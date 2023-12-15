"""
Object tracker class
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import time
from ball_fun.sterio_cameras import SterioCameras
import numpy as np
import ball_fun.ball_detector as ball_detector
import tools.trajectory_fitting as trajectory_fitting
from parameters import Parameters
from ball_fun.the_hooop import HoopClassifier

class PathPoint:
    """
    Point object to store relevant information
    for locations of 3d Paths
    """
    
    def __init__(self, x: int, y: int, z: int, orig: tuple, time_stamp: int) -> None:
        """
        
        """
        self.x = x
        self.y = y
        self.z = z
        self.orig = orig
        self.time_stamp  = time_stamp
        self.in_degree = 0
        self.out_degree = 0

    @staticmethod
    def calc_speed(point1: 'PathPoint', point2: 'PathPoint'):
        """
        Calculate speed between two points

        Arguments:
            point1 : PathPoint object of first point
            point2 : PathPoint object of second point
        """
        if type(point2) is not PathPoint or type(point1) is not PathPoint:
            raise NotImplementedError('Subtractin only supported between PathPoint objects')
        
        change_dist = np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)
        change_time = (point2.time_stamp - point1.time_stamp) / 1e9

        return np.abs(change_dist / change_time)
    
    def __str__(self) -> str:
        return f'X:{self.x} Y:{self.y} Z:{self.z}'
    
    def __repr__(self) -> str:
        return f'P({self.x} {self.y} {self.z})'

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
            if point.out_degree != 0:
                continue
            for loc in new_locs:
                speed = PathPoint.calc_speed(point, loc)
                # print('speed', speed / 1000, 'm/s')

                if self.min_speed <= speed <= self.max_speed:
                    self.adjacency_list[point].add(loc)
                    point.out_degree += 1
                    loc.in_degree += 1


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

                for node in self.adjacency_list[point]:
                    node.in_degree -= 1

                self.adjacency_list.pop(point)
        
        for point in self.adjacency_list.keys():
            self.adjacency_list[point] = self.adjacency_list[point] - removed
            point.out_degree = len(self.adjacency_list[point])

        self.last_clean_time = curr_time

    def get_lines(self) -> list[tuple[tuple[int, int]]]:
        """
        Get all edges of current graph

        Returns:
            list of point pairs (x,y) for start and end pos 
              of the path
        """

        lines = []

        for start_node, end_list in self.adjacency_list.items():

            for node in end_list:
                lines.append( (( start_node.orig[0], start_node.orig[1] ), (node.orig[0], node.orig[1])) )

        return lines
    
    def get_paths(self, start_node: PathPoint = None) -> list[list[PathPoint]]:
        """
        Gathers a list of possible paths for a projectile
        with the current location structures

        Returns:
            (paths) : list of possible PathPoint paths 
        """

        if start_node is None:
            all_paths = []
            for node in self.adjacency_list.keys():
                if node.in_degree == 0:
                    path = self.get_paths(node)
                    all_paths.extend(path)
            return all_paths

        # base condition
        if len(self.adjacency_list[start_node]) == 0:
            return [[start_node]]
    
        all_paths = []
        
        for node in self.adjacency_list[start_node]:
            path = self.get_paths(node)
            all_paths.extend(path)

        for path in all_paths:
            path.append(start_node)

        return all_paths
    
    def path_to_numpy(self, path: list[PathPoint]) -> np.ndarray:
        """
        Converts a Path to numpy array of x,y,z coords

        Arguments:
            path : list of PathPoint forming the path

        Returns:
            (np.ndarray) : paths as ndarray of shape (3,N)
        """

        np_array = np.zeros((3,len(path)))
        
        for idx, point in enumerate(path):

            np_array[0,idx] = point.x
            np_array[1,idx] = point.y
            np_array[2,idx] = point.z

        return np.flip(np_array, axis=1)
        
    
    def get_best_trajectory_pts(self) -> np.ndarray:
        """
        Determines best trajectory to fit to trackings

        Returns:
            pts3d : np.ndarray((3,N))
            trajXY : trajectory for Y
            trajXZ : trajectory for Z
        """

        paths = self.get_paths()

        best_trajXY, best_trajXZ, bestRes, bestPath_pts = None, None, None, None
        for path in paths:
            pts3d = self.path_to_numpy(path)

            if pts3d.shape[1] < 3:
                continue

            trajXY, trajXZ, res = trajectory_fitting.fit_trajectory(pts3d)

            # TODO create heuristic for balancing pathlength and res ... maybe

            if best_trajXY is None:
                best_trajXY, best_trajXZ, bestRes, bestPath_pts = trajXY, trajXZ, res, pts3d

            elif res < bestRes:
                best_trajXY, best_trajXZ, bestRes, bestPath_pts = trajXY, trajXZ, res, pts3d

        if best_trajXY is None:
            return np.ndarray((3,0)), best_trajXY, best_trajXZ
        
        return trajectory_fitting.predict_future_positions(bestPath_pts, best_trajXY, best_trajXZ, 100), best_trajXY, best_trajXZ



def tracking_stream(sterio_pair: SterioCameras) -> None:

    tracking = ProjectileTracking(Parameters.max_time, Parameters.min_speed, Parameters.max_speed)
    detector = ball_detector.HoughBallDetector(2, minRadius=Parameters.minRadius, maxRadius=Parameters.maxRadius)
    # detector = ball_detector.HybridBallDetector(2)
    hoop_pred = HoopClassifier()
    hoop_pred.locate(sterio_pair)

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

        tracking.clean()
        # for line in tracking.get_lines():
        #     cv2.line(imgL, line[0], line[1], (0, 0, 255), 2)

        pts3_traj, trajXY, trajXZ = tracking.get_best_trajectory_pts()

        if trajXY is not None and trajXZ is not None:
            make_prob = hoop_pred.make_prob(trajXY, trajXZ)

            if make_prob > 0.85:
                cv2.putText(imgL, "YIPPIE!!! ", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(imgR, "YIPPIE!!! ", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            elif make_prob > 0.1:
                cv2.putText(imgL, "BOOOO!!! ", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(imgR, "BOOOO!!! ", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

            # only draw trajectories with prob of going in > X
            if make_prob > 0.1:
                pts2LP_traj, pts2RP_traj = sterio_pair.project(pts3_traj)
                for i in range(1, pts2LP_traj.shape[1]):
                    imgL = cv2.line(imgL, (pts2LP_traj[0,i], pts2LP_traj[1,i]), (pts2LP_traj[0,i-1], pts2LP_traj[1,i-1]), (0,0,255),2)
                    imgR = cv2.line(imgR, (pts2RP_traj[0,i], pts2RP_traj[1,i]), (pts2RP_traj[0,i-1], pts2RP_traj[1,i-1]), (0,0,255),2)

  

        return imgL, imgR
    
    sterio_pair.show_stream(0.5, 0.5, clsffy)


if __name__ == '__main__':

    # DEMO
    sterio_pair = SterioCameras(["./media/test_vid_104_L.mp4", "./media/test_vid_105_R.mp4"])

    tracking_stream(sterio_pair)
    sterio_pair.close()