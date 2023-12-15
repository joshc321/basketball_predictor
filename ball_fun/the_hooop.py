"""
The hoop and all
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ball_fun.sterio_cameras import SterioCameras
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import cv2

class HoopClassifier:
    """
    Everything to do with the HOOP
    position estimator for the hoop and
    predictor for if a trajectory will
    go in
    """
    def __init__(self, model_path: str = f'{Path(__file__).parent.parent}/YOLO_model/models/best.pt') -> None:
        """
        Hoop Classifier init
        Arguments:
            model_path : path to YOLO model weigths file
        """
        self.model = YOLO(model_path)
        self.model.to(device='cpu')
        self._hoop_loc = np.array([
            [], # back left front right
            [],
        ])

    def locate(self, sterio_cameras: SterioCameras) -> None:
        """
        Locates the hoop and estimates its position in
        3D space
        """

        foundL, foundR = False, False
        for _ in range(100):
            success, frameL, frameR = sterio_cameras.grab_frame()
            results = self.model.predict([frameL, frameR], verbose=False)

            resultL, resultsR = results[0].to('cpu').numpy(), results[1].to('cpu').numpy()
            for boxL in resultL.boxes:
                clsL = boxL.cls[0]
                if clsL == 1:
                    foundL = True
                    break
            for boxR in resultsR.boxes:
                clsR = boxR.cls[0]
                if clsR == 1:
                    foundR = True
                    break
            
            if foundL and foundR:
                break

        # TODO if time permits automate with YOLO
        # and clean...

        # Accuracy test
        # pointsL = np.array([
        #     [458,456,511,520],
        #     [539,559,566,548]
        # ])
        # pointsR = np.array([
        #     [78,58,112,131],
        #     [509,532,536,516]
        # ])
        # pts3 = sterio_cameras.triangulate(pointsL, pointsR)
        # print(pts3)
        # self._hoop_loc = pts3
        # return 

        plt.imshow(frameL[:,:,[2,1,0]])
        plt.title("Hoop locator")

        pointsL = []
        promp_list = ["Click on the back of the hoop", "Click on the left of the hoop", "Click on the front of the hoop", "Click on the right of the hoop"]
        for i in range(4):
            plt.title(promp_list[i])
            plt.draw()
            clicked_point = plt.ginput(1, show_clicks=True, timeout=0, mouse_add=1)[0]
            pointsL.append(clicked_point)
        
        pointsL = np.array(pointsL).T

        plt.imshow(frameR[:,:,[2,1,0]])
        plt.title("Hoop locator")

        pointsR = []
        for i in range(4):
            plt.title(promp_list[i])
            plt.draw()
            clicked_point = plt.ginput(1, show_clicks=True, timeout=0, mouse_add=1)[0]
            pointsR.append(clicked_point)
        pointsR = np.array(pointsR).T
        pts3 = sterio_cameras.triangulate(pointsL, pointsR)
        plt.close()

        self._hoop_loc = pts3



        

    def make_prob(self, traj_xy, traj_xz) -> float:
        """
        Determines if a given trajectory will go in the hoop
        """

        hoop_center = np.average(self._hoop_loc, axis=1)


        ball_pos = np.array([hoop_center[0], traj_xy(hoop_center[0]), traj_xz(hoop_center[0])])

        dist = np.sum((hoop_center-ball_pos)**2, axis=0)
        dist = np.sqrt(dist)

        hoop_dist_lr = np.sqrt(np.sum((self._hoop_loc[:,1] - self._hoop_loc[:,3])**2, axis=0))
        hoop_dist_fb = np.sqrt(np.sum((self._hoop_loc[:,0] - self._hoop_loc[:,2])**2, axis=0))
        hoop_rad = (hoop_dist_fb + hoop_dist_lr) / 4

        # lib prob
        m = (0 - 1) / (800 - hoop_rad)
        b = 1 - m * hoop_rad  
        prob = m * dist + b
        if prob > 1:
            prob = 1.0
        elif prob < 0:
            prob = 0.0
        return prob