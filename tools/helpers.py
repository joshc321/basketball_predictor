import subprocess
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import os

def get_left_right_camera_idxs(cam_name: str = "C270 HD WEBCAM"):
    """
    Get camera number for left and right cameras. Doen't currently specify
    which camera is on the left and which is on the right

    Arguments: 
        cam_name (str) : name of camera
    
    Returns:
        [int, int] : camera numbers for left and right camera unordered
    """

    if os.name == "nt":
        return [0,1]

    command = ['ffmpeg','-f', 'avfoundation','-list_devices','true','-i','""']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    matching_ids = []

    for item in result.stderr.splitlines():
        if cam_name in item:
            cam_id = int(item.split("[")[2].split(']')[0])
            matching_ids.append(cam_id)

        if "AVFoundation audio devices" in item:
            break

    if len(matching_ids) != 2:
        raise IndexError("Left or Right camera not found")

    return matching_ids

def determine_left_image(img1: np.ndarray, img2: np.ndarray) -> int:
    """
    Determines which input image is the left image from a sterio pair

    Arguments:
        img1 [np.ndarray] : image from camera 1
        img2 [np.ndarray] : image from camera 2

    Returns:
        [0 | 1 | 2] : value of left image
                        0 -> unable to determine
                        1 -> img1 is left image
                        2 -> img2 is left image 
    """


    # based on https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    img1_cnt = 0
    img2_cnt = 0
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.1*n.distance:
            x_img1 = kp1[m.queryIdx].pt[0]
            x_img2 = kp2[m.trainIdx].pt[0]

            if x_img1 > x_img2:
                img1_cnt += 1
            else:
                img2_cnt += 1

    if img1_cnt > img2_cnt:
        return 1
    elif img2_cnt > img1_cnt:
        return 2
    return 0

def triangulate(pts2L: np.ndarray, pts2R: np.ndarray, mtxL: np.ndarray, mtxR: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Sterio camera triangulation
    
    Arguments:
        pts2L : 2D point locations in left camera
        pts2R : 2D point locations in right camera
        mtxL  : Camera matrix of left cam
        mtxR  : Camera matrix of right cam
        R     : Rotation matrix between left and right camera
        T     : Translation matrix between left and right camera

    Returns:
        pts3 : 3D triangulated point locations 
    """


    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtxL @ RT1 # projection matrix for C1

    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtxR @ RT2 # projection matrix for C2

    npts = pts2L.shape[1]
    pts3 = np.zeros((3,npts))

    for i in range(npts):
        A = np.array([
            pts2L[1,i] * P1[2,:] - P1[1,:],
            P1[0,:] - pts2L[0,i] * P1[2,:],
            pts2R[1,i] * P2[2,:] - P2[1,:],
            P2[0,:] - pts2R[0,i] * P2[2,:],
        ])
        B = A.transpose() @ A
        Vh = np.linalg.svd(B, full_matrices=False).Vh

        pts3[:,i] = (Vh[3,0:3]/Vh[3,3]).T

    return pts3


if __name__ == '__main__':
    idxs = get_left_right_camera_idxs()
    print(idxs)