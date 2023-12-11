import abc
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import time
import threading

class BallDetector(abc.ABC):
    """
    Top level class for Ball detection algorithms
    """

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Predicts locations of the ball on input images

        Arguments:
            images : list of N images

        Returns:
            centers : numpy array of coordinates of center locations of found balls
                ->  list of length N with each
                    element having shape (2,P) 
                    with P being the number of
                    points found with type int
        """
        pass

class HoughBallDetector(BallDetector):

    def __init__(self, num_predictors: int, minRadius: int = 20, maxRadius: int = 25) -> None:
        """
        Hough Circle detector ball classifier

        Arguments:
            num_predictors : number of predictors needed to predict on images
                             should be equal to the number of images passed
                             into the predict method
        """
        self.fgbg = []
        for _ in range(num_predictors):
            self.fgbg.append(cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=150, detectShadows=False))

        # HoughCircles params
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.dp = 1
        self.minDist = 20
        self.param1 = 300
        self.param2 = 20

        # concurrent vars (KINDA)
        self.list_lock = threading.Lock()


    def single_predict(self, index, image, fgbg, return_list) -> np.ndarray:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = fgbg.apply(image, -1)
        # ret, motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)

        med_blur = cv2.medianBlur(motion_mask, 5)
        med_gaus_blur = cv2.GaussianBlur(med_blur, (5,5), 0)

        # temp_f = cv2.resize(med_gaus_blur, (frame.shape[1] // 2, frame.shape[0] // 2))
        # cv2.imshow("mask", temp_f)

        masked = med_gaus_blur

        circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, self.dp, self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)

        center = np.ndarray((2,0), dtype=int)

        if circles is not None:
            circles_processed = circles[0,:][:,0:2].T
            center = circles_processed.astype(int)

        with self.list_lock:
            return_list[index] = center

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:

        centers = [[],[]]
        # s = time.time_ns()
        # for idx, frame in enumerate(images):
        #     res = self.single_predict(idx, frame, self.fgbg[idx])
        #     centers.append(res)

        thread_list = []
        for idx, frame in enumerate(images):
            t = threading.Thread(target=self.single_predict, args=(idx, frame, self.fgbg[idx], centers))
            t.start()
            thread_list.append(t)
        
        for t in thread_list:
            t.join()

        # print('prediction took', round((time.time_ns() - s)/1e6 , 2), 'ms')

        return centers
    
class YOLOBallDetector(BallDetector):
    
    def __init__(self) -> None:
        self.model = YOLO(f'{Path(__file__).parent}/best.pt')
        self.model.to(device='cpu')

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:


        s = time.time_ns()

        results = self.model.predict(images, verbose=False)

        centers = []

        for result in results:
            result = result.to('cpu').numpy()
            pts2 = [[],[]]
            for boxL in result.boxes:
            
                xyxy, cls, conf = boxL.xyxy[0], boxL.cls[0], boxL.conf[0]

                if cls != 0:
                    continue
                
                # add x and y coord of center to pts list
                pts2[0].append( int((xyxy[0] + xyxy[2]) / 2) )
                pts2[1].append( int((xyxy[1] + xyxy[3]) / 2) )
            
            centers.append(np.array(pts2))
        
        print('prediction took', round((time.time_ns() - s)/1e6 , 2), 'ms')

        return centers
    