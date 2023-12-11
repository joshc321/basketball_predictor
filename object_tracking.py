import cv2
from ultralytics import YOLO
from pathlib import Path
import tools.helpers as helpers
import collections
import time
from sterio_cameras import SterioCameras
import numpy as np
import matplotlib.pyplot as plt
import ball_detector

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

        self.timeline = []

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
            self.timeline.append(new_loc)

        for point in self.adjacency_list.keys():
            for loc in new_locs:
                speed = PathPoint.calc_speed(point, loc)
                # print('speed', speed / 1000, 'm/s')

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

        for idx, point in enumerate(self.timeline):
            if (curr_time - point.time_stamp) > self.max_time:
                self.timeline = self.timeline[idx:]

        self.last_clean_time = curr_time

    def get_lines(self) -> list[tuple[tuple[int, int]]]:

        lines = []

        # for start_node, end_list in self.adjacency_list.items():

        #     for node in end_list:
        #         lines.append( (( start_node.orig[0], start_node.orig[1] ), (node.orig[0], node.orig[1])) )

        for i in range(1, len(self.timeline)):
            lineL = self.timeline[i - 1]
            lineR = self.timeline[i]

            lines.append( (( lineL.orig[0], lineL.orig[1] ), (lineR.orig[0], lineR.orig[1])) )

        return lines



def tracking_stream(sterio_pair: SterioCameras) -> None:

    tracking = ProjectileTracking(2.0, 1, 10)
    detector = ball_detector.HoughBallDetector(2, minRadius=16, maxRadius=35)
    # pts3_all = [[],[],[]]
    pts3_all = []

    def clsffy(imgL, imgR):

        snapshot_time = time.time_ns()

        result = detector.predict([imgL, imgR])

        result_L = result[0]
        result_R = result[1]

        pts2L, pts2R = [[],[]], [[],[]]
        ptsOrigL = []

        if result_L.shape[1] > 0 and result_R.shape[1] > 0:

            epilines = cv2.computeCorrespondEpilines(result_L.T, 1,sterio_pair.fund_mtx).reshape(-1,3)

            for idx, centerL in enumerate(result_L.T):
                xL, yL = centerL
                # print('L', xL, yL)
                for xR,yR in result_R.T:
                    
                    dist = np.abs(epilines[idx][0] * xR + epilines[idx][1] * yR + epilines[idx][2]) / np.sqrt( epilines[idx][0]**2 + epilines[idx][1]**2 )
                    # print('R', xR ,yR)
                    # print('dist', dist)
                    if dist > 80:
                        continue

                    pts2L[0].append(xL)
                    pts2L[1].append(yL)
                    pts2R[0].append(xR)
                    pts2R[1].append(yR)
                    ptsOrigL.append(centerL)

                    imgL = cv2.circle(imgL,centerL,10,(0,255,0),10)
                    imgR = cv2.circle(imgR,(xR,yR),10,(0,255,0),10)

                
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
        
        # for node, v in tracking.adjacency_list.items():
        #     if len(v) > 0:
        #         pts3_all[0].append(node.x)
        #         pts3_all[1].append(node.y)
        #         pts3_all[2].append(node.z)

        if pts3.shape[1] > 0:
            pts3_all.append(pts3)
        #     print(pts3)

        # imgL = result_L.plot()
        return imgL, imgR
    
    sterio_pair.show_stream(0.5, 0.5, clsffy)

    # pts3_np = np.array(pts3_all)
    pts3_np = np.zeros((3,1))
    for pts3 in pts3_all:
        pts3_np = np.hstack((pts3_np, pts3))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pts3_np[0,:],pts3_np[1,:],pts3_np[2,:],'.')
    plt.show()

# model = YOLO(f'{Path(__file__).parent}/best.pt')
# model.to(device='cpu')
sterio_pair = SterioCameras(["./media/test_vid_104_L.mp4", "./media/test_vid_105_R.mp4"])
# sterio_pair = SterioCameras()
ball_centers = collections.deque(maxlen=20)

tracking_stream(sterio_pair)
sterio_pair.close()
