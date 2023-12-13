"""
script to run background
subtraction and visulization
for debugging
"""

import cv2
import numpy as np

# Open the video file
# cap = cv2.VideoCapture(helpers.get_left_right_camera_idxs()[0])
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./media/test_vid_104_L.mp4")


fgbg = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=150, detectShadows=False)


timer = cv2.getTickCount()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    

    if success:

        # image base transforms
        frame = cv2.flip(frame, -1)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = fgbg.apply(frame, -1)
        # ret, motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)

        med_blur = cv2.medianBlur(motion_mask, 5)
        med_gaus_blur = cv2.GaussianBlur(med_blur, (5,5), 0)

        # temp_f = cv2.resize(med_gaus_blur, (frame.shape[1] // 2, frame.shape[0] // 2))
        # cv2.imshow("mask", temp_f)

        masked = med_gaus_blur

        circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, 1, 100, param1=300, param2=20, minRadius=30, maxRadius=55)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                print('radius', i[2])
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)


        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display the annotated frame
        cv2.imshow("frame", frame)
        cv2.imshow("Motion Mask", med_gaus_blur)
        timer = cv2.getTickCount()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
