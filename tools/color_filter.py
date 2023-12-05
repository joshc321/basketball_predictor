import cv2
import numpy as np
import tools.helpers as helpers

cap = cv2.VideoCapture(helpers.get_left_right_camera_idxs()[1])

hue_slider_max = 360
sat_slider_max = 255
val_slider_max = 255

color_lower = np.array([9, 105, 72])
color_upper = np.array([20, 255, 255])

def low_hue(val):
    color_lower[0] = val
def low_sat(val):
    color_lower[1] = val
def low_val(val):
    color_lower[2] = val
def high_hue(val):
    color_upper[0] = val
def high_sat(val):
    color_upper[1] = val
def high_val(val):
    color_upper[2] = val

cv2.namedWindow("the title")
cv2.createTrackbar("low hue", "the title" , 0, hue_slider_max, low_hue)
cv2.createTrackbar("low sat", "the title" , 0, sat_slider_max, low_sat)
cv2.createTrackbar("low val", "the title" , 0, val_slider_max, low_val)
cv2.createTrackbar("high hue", "the title" , 0, hue_slider_max, high_hue)
cv2.createTrackbar("high sat", "the title" , 0, sat_slider_max, high_sat)
cv2.createTrackbar("high val", "the title" , 0, val_slider_max, high_val)


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.flip(frame, 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)

        # Display the annotated frame
        cv2.imshow("Frame", mask)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


