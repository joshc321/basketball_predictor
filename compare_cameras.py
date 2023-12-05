import cv2
from pathlib import Path
import tools.helpers as helpers


# model = YOLO(f'{Path(__file__).parent}/best.pt')

cap1 = cv2.VideoCapture(f'{Path(__file__).parent}/out_100.mp4', apiPreference=cv2.CAP_AVFOUNDATION)
cap2 = cv2.VideoCapture(f'{Path(__file__).parent}/out_200.mp4', apiPreference=cv2.CAP_AVFOUNDATION)


while cap1.isOpened() and cap2.isOpened():
    # Read a frame from the video

    success1, frame = cap1.read()
    success2, frame2 = cap2.read()

    if success1 and success2:

        f1_h, f1_w, _ = frame.shape
        f2_h, f2_w, _ = frame2.shape


        frame = cv2.flip(frame, 0)
        frame2 = cv2.flip(frame2, 0)


        frame = cv2.resize(frame, (f1_w // 2, f1_h // 2))
        frame2 = cv2.resize(frame2, (f2_w // 2, f2_h // 2))
        

        # Display the frame

        cv2.imshow("im1", frame)
        cv2.imshow("im2", frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap1.release()
cap2.release()
cv2.destroyAllWindows()