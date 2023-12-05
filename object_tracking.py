import cv2
from ultralytics import YOLO
from pathlib import Path
import tools.helpers as helpers
import collections
import time
from sterio_cameras import SterioCameras
import numpy as np

class PathPoint:
    
    def __init__(self, x: int, y: int, z: int, orig: tuple, time_stamp: int) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.orig = orig
        self.time_stamp  = time_stamp

    @staticmethod
    def calc_speed(left: 'PathPoint', right: 'PathPoint'):
        if type(right) is not PathPoint or type(left) is not PathPoint:
            raise NotImplementedError('Subtractin only supported between PathPoint objects')
        
        change_dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2 + (left.z - right.z)**2)
        change_time = (right.time_stamp - left.time_stamp) / 1e9

        return np.abs(change_dist / change_time)

class ProjectileTracking:

    def __init__(self, max_time: int, min_speed: float, max_speed: float) -> None:
        """
        projectile tracking class

        Arguments:
            max_time (int) : max time allowed keep a tracking location in seconds
            min_speed (float) : min speed allowed when connecting points m/s
            max_speed (float) : max speed allowed when connecting points m/s
        """
        self.max_time = max_time * 1e9 # convert to ns
        self.min_speed = min_speed * 1000 # convert to mm/s
        self.max_speed = max_speed * 1000 # convert to mm/s

        self.last_clean_time = time.time_ns()

        self.adjacency_list = {}

    def add_locations(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ptsOrigL: list, time_stamp: int) -> None:
        """
        Add a point to be tracked
        
        Arguments:
            x (ndarray) : x coordinates of item
            y (ndarray) : y coordinates of item
            z (ndarray) : z coordinates of item
            time_stamp (int) : time in ns of item capture
        """

        new_locs = []
        for i in range(len(x)):
            new_loc = PathPoint(x[i],y[i],z[i],ptsOrigL[i],time_stamp)
            new_locs.append(new_loc)

        for point in self.adjacency_list.keys():
            for loc in new_locs:
                speed = PathPoint.calc_speed(point, loc)
                print('speed', speed / 1000, 'm/s')

                if self.min_speed <= speed <= self.max_speed:
                    self.adjacency_list[point].add(new_loc)

        for new_loc in new_locs:
            self.adjacency_list[new_loc] = set()

    def clean(self) -> None:
        """
        Removes points from the tracking that is older than
        a maximum age
        """
        curr_time = time.time_ns()

        if curr_time - self.last_clean_time < self.max_time:
            return

        removed = set()

        for point in list(self.adjacency_list.keys()):
            if (curr_time - point.time_stamp) > self.max_time:
                removed.add(point)
                self.adjacency_list.pop(point)
        
        for point in self.adjacency_list.keys():
            self.adjacency_list[point] = self.adjacency_list[point] - removed

        self.last_clean_time = curr_time

    def get_lines(self) -> list[tuple[tuple[int, int]]]:

        lines = []

        for start_node, end_list in self.adjacency_list.items():

            for node in end_list:
                lines.append( (( start_node.orig[0], start_node.orig[1] ), (node.orig[0], node.orig[1])) )

        return lines



model = YOLO(f'{Path(__file__).parent}/best-mps.pt')
model.to(device='mps')
sterio_pair = SterioCameras()

ball_centers = collections.deque(maxlen=20)


def tracking_stream(sterio_pair: SterioCameras, classifier: YOLO) -> None:

    tracking = ProjectileTracking(2.0, 1.5, 3.5)


    def clsffy(imgL, imgR):

        # result = classifier.predict([imgL, imgR], iou=0.7, conf=0.75)
        # result = classifier1.predict(imgL)
        # result1 = classifier2.track(imgR, persist=True, tracker="bytetrack.yaml")
        snapshot_time = time.time_ns()

        result = classifier.predict([imgL, imgR], verbose=False)

        result_L = result[0].to('cpu').numpy()
        result_R = result[1].to('cpu').numpy()
        imgL = result_L.plot()
        imgR = result_R.plot()

        pts2L, pts2R = [[],[]], [[],[]]
        ptsOrigL = []

        for boxL in result_L.boxes:
            
            xyxyL, clsL, confL = boxL.xyxy[0], boxL.cls[0], boxL.conf[0]

            if clsL != 0:
                continue

            centerL = (int((xyxyL[0] + xyxyL[2]) / 2), int((xyxyL[1] + xyxyL[3]) / 2))

            for boxR in result_R.boxes:
                xyxyR, clsR, confR = boxR.xyxy[0], boxR.cls[0], boxR.conf[0]
                
                if clsR != 0:
                    continue

                centerR = (int((xyxyR[0] + xyxyR[2]) / 2), int((xyxyR[1] + xyxyR[3]) / 2))

                pts2L[0].append(centerL[0])
                pts2L[1].append(centerL[1])
                pts2R[0].append(centerR[0])
                pts2R[1].append(centerR[1])
                ptsOrigL.append(centerL)
            
        pts3 = sterio_pair.triangulate(np.array(pts2L), np.array(pts2R))
        tracking.add_locations(pts3[0,:], pts3[1,:], pts3[2,:], ptsOrigL, snapshot_time)

        # for i in range(1, len(ball_centers)):
        #     if ball_centers[i - 1] is None or ball_centers[i] is None:
        #         continue
        #     thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
        #     cv2.line(imgL, ball_centers[i - 1], ball_centers[i], (0, 0, 255), thickness)

        tracking.clean()
        for line in tracking.get_lines():
            cv2.line(imgL, line[0], line[1], (0, 0, 255), 2)

        # imgL = result_L.plot()
        return imgL, imgR
    
    sterio_pair.show_stream(0.5, 0.5, clsffy)


tracking_stream(sterio_pair, model)
sterio_pair.close()
