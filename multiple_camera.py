import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

while cap1.isOpened() and cap2.isOpened():
    # Read a frame from the video
    success1, frame = cap1.read()
    success2, frame2 = cap2.read()

    f1_h, f1_w, _ = frame.shape
    f2_h, f2_w, _ = frame2.shape


    frame = cv2.resize(frame, (f1_w // 2, f1_h // 2))
    frame2 = cv2.resize(frame2, (f2_w // 2, f2_h // 2))


    if success1 and success2:

        # Display the annotated frame
        cv2.imshow("im1", frame)
        cv2.imshow("im2", frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap1.release()
cap2.release()
cv2.destroyAllWindows()