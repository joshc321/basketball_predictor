"""
Notes:
    When gathering calibration images and running calibration
    ensure checker board pattern is detected in same order
    in both left and right images otherwise it will have
    a high RMS value.

"""

import numpy as np
import cv2
import pickle
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from sterio_cameras import LogitechC270, SterioCameras
except ImportError as e:
    print('Unable to import sterio_cameras', e)


MIN_IMAGES = 15 # number of images to gather for calibration
ROWS = 8 # number of checkerboard rows
COLUMNS = 6 # number of checkerboard cols
WORLD_SCALING = 27.4 # the real world square size in mm

def find_chess_board_corners(img: np.ndarray):
    """
    Finds chess board corners of an image without altering the image

    Arguments: 
        img (np.ndarray) : image to find chess board pattern on as numpy array

    Returns:
        (bool, np.ndarray, np.ndarray) : 
                            bool : true if success
                            np.ndarray : corners 2d points in image plane
                            np.ndarray : object point in real world space
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ROWS*COLUMNS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLUMNS].T.reshape(-1,2)
    objp = WORLD_SCALING * objp

    # Arrays to store object points and image points from all the images.
    objpoints = None # 3d point in real world space
    imgpoints = None # 2d points in image plane.
    ret = False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (ROWS,COLUMNS), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints = objp
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints = corners2
    
    return ret, imgpoints, objpoints

def gather_sterio_calibration_images(sterio_camera: 'SterioCameras', output_dir: Path, min_images: int = MIN_IMAGES):
    output_dir.mkdir(exist_ok=True, parents=True)
    images = []
    img_descriptor = 0
    while (len(images) // 2) < min_images:

        ret, imgL, imgR = sterio_camera.grab_frame()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
         
        orig_imgL = imgL.copy()
        orig_imgR = imgR.copy()

        # Find the chess board corners
        retL, cornersL, _ = find_chess_board_corners(imgL)
        retR, cornersR, _ = find_chess_board_corners(imgR)

        # If found, add object points, image points (after refining them)
        if retL == True and retR == True:
            # Draw and display the corners
            img_pathL = output_dir / f'checkerboard_{img_descriptor}_L.png'
            img_pathR = output_dir / f'checkerboard_{img_descriptor}_R.png'
            while img_pathL.exists() or img_pathR.exists():
                img_descriptor += 1
                img_pathL = output_dir / f'checkerboard_{img_descriptor}_L.png'
                img_pathR = output_dir / f'checkerboard_{img_descriptor}_R.png'

            cv2.drawChessboardCorners(imgL, (ROWS,COLUMNS), cornersL, ret)
            cv2.drawChessboardCorners(imgR, (ROWS,COLUMNS), cornersR, ret)
            cv2.imshow('imgL', imgL)
            cv2.imshow('imgR', imgR)
            if cv2.waitKey(0) & 0xFF == ord("y"):
                img_pathL = str(img_pathL)
                img_pathR = str(img_pathR)
                cv2.imwrite(img_pathL, orig_imgL)
                cv2.imwrite(img_pathR, orig_imgR)
                images.append(img_pathL)
                images.append(img_pathR)
        else:
            cv2.imshow('imgL', imgL)
            cv2.imshow('imgR', imgR)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    sterio_camera.close()
    cv2.destroyAllWindows()

    return images

def gather_calibration_images(camera: 'LogitechC270', output_dir: Path, min_images: int = MIN_IMAGES):

    output_dir.mkdir(exist_ok=True, parents=True)

    images = []
    img_descriptor = 0

    while len(images) < min_images:

        ret, img = camera.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
         
        orig_img = img.copy()

        # Find the chess board corners
        ret, corners, _ = find_chess_board_corners(img)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            # Draw and display the corners
            img_path = output_dir / f'checkerboard_{img_descriptor}.png'
            while img_path.exists():
                img_descriptor += 1
                img_path = output_dir / f'checkerboard_{img_descriptor}.png'

            cv2.drawChessboardCorners(img, (ROWS,COLUMNS), corners, ret)
            cv2.imshow('img', img)
            if cv2.waitKey(0) & 0xFF == ord("y"):
                img_path = str(img_path)
                cv2.imwrite(img_path, orig_img)
                images.append(img_path)
        else:
            cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    camera.close()
    cv2.destroyAllWindows()

    return images

def calibrate_camera(image_path: Path, display: bool = False):
    """
    
    """

    images_names = sorted(image_path.rglob('*.png'))

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ROWS*COLUMNS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLUMNS].T.reshape(-1,2)
    objp = WORLD_SCALING * objp

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for img_name in images_names:


        img = cv2.imread(str(img_name))
        img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ROWS,COLUMNS), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (ROWS,COLUMNS), corners2, ret)
        else:
            print('checkerboard pattern not found for', img_name)
        
        if display:
            cv2.imshow('img', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


    # now perform the calibration
    print('calibrating on', len(imgpoints), 'images')
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    print('RMSE ideal < 1: ', ret)

    return K, dist

def stereo_calibrate(mtxL, distL, mtxR, distR, image_path: Path, display: bool = False):
    #read the synched frames
    left_image_names = sorted(image_path.glob('*_L.png'))
    right_image_names = sorted(image_path.glob('*_R.png'))
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((ROWS*COLUMNS,3), np.float32)
    objp[:,:2] = np.mgrid[0:ROWS,0:COLUMNS].T.reshape(-1,2)
    objp = WORLD_SCALING * objp
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for imgL_name, imgR_name in zip(left_image_names, right_image_names):
        # print('searching on', imgL_name.name, imgR_name.name)

        imgL = cv2.imread(str(imgL_name))
        imgR = cv2.imread(str(imgR_name))

        img_size = (imgL.shape[1], imgL.shape[0])

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (ROWS,COLUMNS), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (ROWS,COLUMNS), None)
 
        if retL == True and retR == True:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(imgL, (ROWS,COLUMNS), cornersL, retL) 
            cv2.drawChessboardCorners(imgR, (ROWS,COLUMNS), cornersR, retR)
            if display:
                cv2.imshow('imgL', imgL)
                cv2.imshow('imgR', imgR)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
 
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

    print('calibrating on', len(objpoints), 'images')
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtxL, distL,
                                                                 mtxR, distR, img_size, criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T, F

def display_and_save_calibration_info(save_file, K, dist):
    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated radial distortion coefficients")
    print(dist)
    print("Individual intrinsic parameters")
    print("fx = ",K[0][0])
    print("fy = ",K[1][1])
    print("cx = ",K[0][2])
    print("cy = ",K[1][2])


    # save the results out to a file for later use
    calib = {}
    calib["fx"] = K[0][0]
    calib["fy"] = K[1][1]
    calib["cx"] = K[0][2]
    calib["cy"] = K[1][2]
    calib["dist"] = dist
    fid = open(save_file, "wb" ) 
    pickle.dump(calib,fid)
    fid.close()

def triangulate(mtx1, mtx2, R, T):
    # https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    pts2L = np.array([
        [651,783,654,787],
        [328,328,433,431]
    ])

    pts2R = np.array([
        [161, 301, 163, 304],
        [300, 298, 405, 401]
    ])


    frame1 = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_L.png')
    frame2 = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_R.png')

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(frame1[:,:,[2,1,0]]) # BGR -> RGB
    ax.scatter(pts2L[0], pts2L[1])
    plt.title("Left img")

    ax = fig.add_subplot(1,2,2)
    ax.imshow(frame2[:,:,[2,1,0]])
    ax.scatter(pts2R[0], pts2R[1])
    plt.title("Right img")
    plt.show()

    t = time.time()
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1

    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2

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
        
    print(time.time() - t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pts3[0,:],pts3[1,:],pts3[2,:],'.')
    plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    _,c,_ = img1.shape
    for i in range(pts1.shape[1]):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -lines[i][2]/lines[i][1] ])
        x1,y1 = map(int, [c, -(lines[i][2]+lines[i][0]*c)/lines[i][1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pts1[:,i].flatten()),5,color,-1)
        img2 = cv2.circle(img2,tuple(pts2[:,i].flatten()),5,color,-1)
    return img1,img2

def draw_epilines(F):

    imgL = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_L.png')
    imgR = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_R.png')

    pts2L = np.array([
        [651,783,654,787],
        [328,328,433,431]
    ])

    pts2R = np.array([
        [161, 301, 163, 304],
        [300, 298, 405, 401]
    ])

    lines1 = cv2.computeCorrespondEpilines(pts2L.T, 1,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(imgR,imgL,lines1,pts2R,pts2L)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts2R.T, 2,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(imgL,imgR,lines2,pts2L,pts2R)
    plt.subplot(121),plt.imshow(img5[:,:,[2,1,0]])
    plt.subplot(122),plt.imshow(img3[:,:,[2,1,0]])
    plt.show()

def rectify_images(K1, D1, K2, D2, R, T):
    pts2L = np.array([
        [651,783,654,787],
        [328,328,433,431]
    ])

    pts2R = np.array([
        [161, 301, 163, 304],
        [300, 298, 405, 401]
    ])

    imgL = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_L.png')
    imgR = cv2.imread('./tools/camera_calibration/imgs/synced/checkerboard_3_R.png')

    # Stereo rectification
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, imgL.shape[0:2][::-1], R, T)

    
    s = time.time()
    map1, map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imgL.shape[0:2][::-1], cv2.CV_16SC2)
    img_left_rect = cv2.remap(imgL, map1, map2, cv2.INTER_LINEAR)
    
    map1, map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imgR.shape[0:2][::-1], cv2.CV_16SC2)
    img_right_rect = cv2.remap(imgR, map1, map2, cv2.INTER_LINEAR)
    print(time.time() - s)
    

    plt.subplot(221),plt.imshow(img_left_rect[:,:,[2,1,0]])
    plt.subplot(222),plt.imshow(img_right_rect[:,:,[2,1,0]])
    plt.subplot(223),plt.imshow(imgL[:,:,[2,1,0]])
    plt.subplot(224),plt.imshow(imgR[:,:,[2,1,0]])
    plt.show()

    return img_left_rect, img_right_rect


if __name__ == '__main__':

    # camera setup
    # cap = LogitechC270(1)
    # if not cap.is_opened():
    #     print("cannot open camera")
    #     exit()

    # imgs = gather_calibration_images(cap, Path('./checker_tmp'))

    mtxL, distL = calibrate_camera(Path('./tools/camera_calibration/imgs/left_cam'))
    mtxR, distR = calibrate_camera(Path('./tools/camera_calibration/imgs/right_cam'))
    # display_and_save_calibration_info('calibration.pickle', K, dist)

    # imgs = gather_sterio_calibration_images(SterioCameras(), Path('./checker_tmp'))

    R, T, F = stereo_calibrate(mtxL, distL, mtxR, distR, Path('./tools/camera_calibration/imgs/synced'))
    # print('R\n', R)
    # print('T\n', T)

    # rectify_images(mtxL, distL, mtxR, distR, R, T)

    # draw_epilines(F)

    triangulate(mtxL, mtxR, R, T)