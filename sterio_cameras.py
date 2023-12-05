import cv2
import tools.helpers as helpers
import numpy as np
from typing import Callable
import torch
import time
import ultralytics
from matplotlib import pyplot as plt
import tools.camera_calibration.camera_calibration as camera_calibration
from pathlib import Path
import os


class LogitechC270:
    """
    Camera class for the Logitech C270
    USB webcam
    """
    def __init__(self, cam_id: str | int) -> None:
        self.setup(cam_id)
    
    def setup(self, cam_id: str | int) -> None:
        """
        Setup sequence for camera
        """
        self.start_camera(cam_id)

    def close(self) -> None:
        """
        closing squence for camera
        """
        self._cap.release()
    
    def calibrate_camera(self, calibration_images: str):
        self.mtx, self.dist = camera_calibration.calibrate_camera(Path(calibration_images))

    def start_camera(self, cam_id: str | int) -> None:
        """
        Initialize webcam and store properties
        """

        if os.name == 'nt':
            self._cap = cv2.VideoCapture(cam_id)
        else:
            self._cap = cv2.VideoCapture(cam_id, apiPreference=cv2.CAP_AVFOUNDATION)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'avc1'))

        
        self._width, self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # wait for cameras to start
        if type(cam_id) is int:
            for _ in range(100):
                success, img1 = self.read()
                if success == False or (np.average(img1) > 0):
                    break

    def base_transform_img(self, image: np.ndarray) -> np.ndarray:
        """
        Performs a base image transformation on the read
        image from the camera if any alterations are needed
        before processing

        Arguments:
            image : [np.ndarray] base image
        
        Returns:
            [np.ndarray] : [image] tranformed image
        """

        image = cv2.flip(image, -1)
        return image

    def read(self) -> [bool, np.ndarray]:
        """
        Grabs, decodes and returns the next video frame.

        Returns:
            [bool, np.ndarray] : [success, image]
        """

        success, image = self._cap.read()

        if success:
            # flip image to account for mounted camera
            image = self.base_transform_img(image)
        return success, image
    
    def grab(self) -> bool:
        """
        Grabs the next frame from video file or capturing device.

        Returns:
            [bool] : true if success
        """
        return self._cap.grab()
    
    def retrieve(self) -> [bool, np.ndarray]:
        """
        Decodes and returns the grabbed video frame
        """
        success, image = self._cap.retrieve()

        if success:
            # flip image to account for mounted camera
            image = self.base_transform_img(image)
        return success, image
    
    def get_dimensions(self) -> [int, int]:
        """
        Get dimension of camera image

        Returns:
            [int, int] : [width, height] of image
        """

        return self._width, self._height
    
    def is_opened(self) -> bool:
        """
        Returns true if video capturing has been initialized already

        Returns:
            [bool] : true if video capture is initialized
        """
        return self._cap.isOpened()
    

class SterioCameras:
    """
    Sterio Camera Class to handle all logic associated with
    sterio camera image retrieval

    Arguments:
        vid_stream : optional path to video stream for videos
    """
    def __init__(self, vid_stream: list[str] = None) -> None:
        
        self.setup(vid_stream)

    def setup(self, vid_stream: list[str] = None) -> None:
        self.start_cameras(vid_stream)
        self.calibrate_cameras()

    def close(self) -> None:
        """
        Close any opened connection and any other
        needed clean up
        """
        self._left_cam.close()
        self._right_cam.close()

    def calibrate_cameras(self):
        self._left_cam.calibrate_camera('./tools/camera_calibration/imgs/left_cam')
        self._right_cam.calibrate_camera('./tools/camera_calibration/imgs/right_cam')

        self.rot_mtx, self.tran_mtx, self.fund_mtx = camera_calibration.stereo_calibrate(
            self._left_cam.mtx, 
            self._left_cam.dist, 
            self._right_cam.mtx, 
            self._right_cam.dist, 
            Path('./tools/camera_calibration/imgs/synced'))
        


    def start_cameras(self, vid_stream: list[str] = None) -> None:
        """
        Initialize the left and right cameras
        """
        assert vid_stream is None or len(vid_stream) == 2, 'Must have two video paths in vid stream'

        if vid_stream is None:
            cam_idx_1, cam_idx_2 = helpers.get_left_right_camera_idxs()
        else:
            cam_idx_1, cam_idx_2 = vid_stream

        #TODO detect which is left and right camera
        self._left_cam = LogitechC270(cam_idx_1)
        self._right_cam = LogitechC270(cam_idx_2)

        success, img1, img2 = self.grab_frame()

        if success == False:
            raise RuntimeError('Unable to start cameras')

        left_img = helpers.determine_left_image(img1, img2)

        if left_img == 2:
            self._left_cam, self._right_cam = self._right_cam, self._left_cam

    def grab_frame(self) -> tuple[bool, np.ndarray, np.ndarray]:
        """
        Grabs a frame from the left and right camera

        Returns:
            [bool, np.ndarray, np.ndarry] : [success, left_image, right_image]
        """
        
        self._left_cam.grab()
        self._right_cam.grab()

        successL, imgL = self._left_cam.retrieve()
        successR, imgR = self._right_cam.retrieve()

        return (successL and successR, imgL, imgR)

    def is_opened(self) -> bool:
        """
        Returns true if video capturing has been initialized already

        Returns:
            [bool] : true if video capture is initialized
        """
        return self._left_cam.is_opened() and self._right_cam.is_opened()
  
    def show_stream(self, width_scale: float = 1.0, height_scale: float = 1.0, process_func: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Displays raw video stream from left and right camera

        Arguments:
            width_scale : [float] amount to scale width by
            height_scale : [float] amount to scale width by
            process_func : function used to process left and right frames
        """


        l_w, l_h = self._left_cam.get_dimensions()
        r_w, r_h = self._right_cam.get_dimensions()

        dimsL_scaled = (int(l_w * width_scale), int(l_h * height_scale))
        dimsR_scaled = (int(r_w * width_scale), int(r_h * height_scale))

        timer = cv2.getTickCount()


        while self._left_cam.is_opened() and self._right_cam.is_opened():

            success, imgL, imgR = self.grab_frame()

            if success:

                # Start timer

                if process_func is not None:
                    imgL, imgR = process_func(imgL, imgR)


                if width_scale != 1.0 and height_scale != 1.0:
                    imgL = cv2.resize(imgL, dimsL_scaled)
                    imgR = cv2.resize(imgR, dimsR_scaled)

                # calculate fps
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                cv2.putText(imgL, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                cv2.putText(imgR, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);


                cv2.imshow("left img", imgL)
                cv2.imshow("right img", imgR)
                timer = cv2.getTickCount()

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        cv2.destroyAllWindows()


    def show_classification_stream(self, classifier: ultralytics.YOLO) -> None:

        def clsffy(imgL, imgR):

            result = classifier.predict([imgL, imgR], iou=0.7, conf=0.75)
            imgL = result[0].to('cpu').plot()
            imgR = result[1].to('cpu').plot()
            return imgL, imgR
        
        self.show_stream(0.5, 0.5, clsffy)

    def triangulate(self, pts2L, pts2R) -> np.ndarray:

        return helpers.triangulate(pts2L, pts2R, self._left_cam.mtx, self._right_cam.mtx, self.rot_mtx, self.tran_mtx)

if __name__ == '__main__':

    sterio_pair = SterioCameras()
    sterio_pair.show_stream()
    sterio_pair.close()