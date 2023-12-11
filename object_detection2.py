import cv2
from pathlib import Path
from collections import deque
import tools.helpers as helpers
import time
import ball_detector

# Open the video file
# cap = cv2.VideoCapture(helpers.get_left_right_camera_idxs()[0])
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./media/test_vid_104_L.mp4")


# fgbg = cv2.createBackgroundSubtractorMOG2()
# detector = ball_detector.HoughBallDetector(1, minRadius=16, maxRadius=35)
# detector = ball_detector.YOLOBallDetector()
detector = ball_detector.HybridBallDetector(1)


timer = cv2.getTickCount()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    

    if success:

        # image base transforms
        frame = cv2.flip(frame, -1)

        centers = detector.predict([frame])[0]
        
        for center in centers.T:
            frame = cv2.circle(frame,center,10,(0,255,0),10)

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display the annotated frame
        cv2.imshow("frame", frame)
        # cv2.imshow("Motion Mask", med_gaus_blur)
        timer = cv2.getTickCount()

        # Break the loop if 'q' is pressed
        # if centers.shape[1] > 0 and cv2.waitKey(0) & 0xFF == ord("q"):
        #     break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
