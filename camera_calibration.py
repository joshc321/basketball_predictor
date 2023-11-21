import numpy as np
import cv2 as cv
import pickle

MIN_IMAGES = 2
resultfile = 'calibration.pickle'

# camera setup
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open camera")
    exit()



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while len(objpoints) < MIN_IMAGES:


    ret, img = cap.read()
    img_size = (img.shape[1], img.shape[0])
    print('img size', img_size, 'ret', ret)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        cv.imshow('img', img)
        if cv.waitKey(0) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()


# now perform the calibration
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size,None,None)

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
fid = open(resultfile, "wb" ) 
pickle.dump(calib,fid)
fid.close()

print("rotations\n", rvecs)
print("translations\n", tvecs)