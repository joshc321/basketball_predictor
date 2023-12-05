import cv2
from ultralytics import YOLO
from pathlib import Path
import tools.helpers as helpers

# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
model = YOLO(f'{Path(__file__).parent}/best.pt')
model2 = YOLO(f'{Path(__file__).parent}/yolo-11-26.pt')



# Open the video file
# cap = cv2.VideoCapture(helpers.get_left_right_camera_idxs()[0])
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("shooting_ball0.mov")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # frame = cv2.flip(frame, 0)

    if success:
        # Run YOLOv8 inference on the frame
        results1 = model(frame)
        results2 = model2(frame)

        # Visualize the results on the frame
        annotated_frame = results1[0].plot()
        annotated_frame2 = results2[0].plot()

        # Display the annotated frame
        cv2.imshow("Model 1", annotated_frame)
        cv2.imshow("model 2", annotated_frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
