import cv2
from pathlib import Path
from collections import deque
import tools.helpers as helpers

# Open the video file
# cap = cv2.VideoCapture(helpers.get_left_right_camera_idxs()[0])
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("shooting_ball0.mov")


fgbg = cv2.createBackgroundSubtractorMOG2()

pts = deque(maxlen=20)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        motion_mask = fgbg.apply(frame, -1)


        circles = cv2.HoughCircles(motion_mask, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=40, minRadius=40, maxRadius=100)
        if circles is not None:
            for i in circles[0,:]:
                frame = cv2.circle(frame,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),1)

        # Display the annotated frame
        # cv2.imshow("Motion Mask", motion_mask)
        cv2.imshow("normal", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
