import cv2
from ultralytics import YOLO
from pathlib import Path
import tools.helpers as helpers
import threading
import queue
import time

# Function to read frames from the webcam
def read_frames(cap, cap2, frame_queue, frame_queue2, stop_event, disp_queue, disp_queue2):
    while not stop_event.is_set():
        cap.grab()
        cap2.grab()

        ret, frame = cap.retrieve()
        ret2, frame2 = cap2.retrieve()
        
        if not ret or not ret2:
            break
        frame_queue.put(frame)
        frame_queue2.put(frame2)

        # flip frames upright to display
        frame = cv2.flip(frame, -1)
        frame2 = cv2.flip(frame2, -1)
        disp_queue.put(frame)
        disp_queue2.put(frame2)


    cap.release()
    cap2.release()

# Function to write frames to a video file
def write_frames(output_file, output_file2, frame_queue, frame_queue2, stop_event, img_size):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_file, fourcc, 30, img_size)  # Adjust parameters as needed
    out2 = cv2.VideoWriter(output_file2, fourcc, 30, img_size)

    while not stop_event.is_set() or not frame_queue.empty() or not frame_queue2.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            out.write(frame)

        if not frame_queue2.empty():
            frame = frame_queue2.get()
            out2.write(frame)

    out.release()
    out2.release()

def start_cameras():
    cam_idx_1, cam_idx_2 = helpers.get_left_right_camera_idxs()
    cap1 = cv2.VideoCapture(cam_idx_1, apiPreference=cv2.CAP_AVFOUNDATION)
    cap2 = cv2.VideoCapture(cam_idx_2, apiPreference=cv2.CAP_AVFOUNDATION)
    print(cap1.getBackendName(), cap2.getBackendName())

    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'avc1'))
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'avc1'))

    # set params
    # cap1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # cap2.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # cap1.set(cv2.CAP_PROP_FPS, 30)
    # cap2.set(cv2.CAP_PROP_FPS, 30)

    return cap1, cap2


# model = YOLO(f'{Path(__file__).parent}/best.pt')

cap1, cap2 = start_cameras()

# Create a queue for frames
frame_queue = queue.Queue()
frame_queue2 = queue.Queue()
disp_queue = queue.Queue()
disp_queue2 = queue.Queue()

# Set up threading events
stop_event = threading.Event()

 # Create and start the threads
read_thread = threading.Thread(
        target=read_frames, 
        args=(cap1, cap2, 
              frame_queue, frame_queue2, 
              stop_event, disp_queue, disp_queue2))
write_thread = threading.Thread(
    target=write_frames, 
    args=(
        f'{Path(__file__).parent}/out_100.mp4',
        f'{Path(__file__).parent}/out_200.mp4', 
        frame_queue, frame_queue2, 
        stop_event, 
        (int(cap1.get(3)), int(cap1.get(4))),
    ))
print('starting thread')
read_thread.start()
write_thread.start()

while True:

    frame = disp_queue.get()
    frame2 = disp_queue2.get()

    f1_h, f1_w, _ = frame.shape
    frame = cv2.resize(frame, (f1_w // 2, f1_h // 2))
    cv2.imshow('f1', frame)
    f2_h, f2_w, _ = frame2.shape
    frame2 = cv2.resize(frame2, (f2_w // 2, f2_h // 2))
    cv2.imshow('f2', frame2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



print('ending')
stop_event.set()
read_thread.join()
write_thread.join()
cv2.destroyAllWindows()
exit()


while cap1.isOpened() and cap2.isOpened():
    # Read a frame from the video

    # cap1.grab()
    # cap2.grab()

    # success1, frame = cap1.retrieve()
    # success2, frame2 = cap2.retrieve()

    success1, frame = cap1.read()
    success2, frame2 = cap2.read()


    if success1 and success2:

        f1_h, f1_w, _ = frame.shape
        f2_h, f2_w, _ = frame2.shape


        frame = cv2.flip(frame, 0)
        frame2 = cv2.flip(frame2, 0)

        # frame = cv2.GaussianBlur(frame, (17,17), 0)
        # frame2 = cv2.GaussianBlur(frame2, (17,17), 0)


        # frame = cv2.resize(frame, (f1_w // 2, f1_h // 2))
        # frame2 = cv2.resize(frame2, (f2_w // 2, f2_h // 2))

        # result1 = model(frame)
        # result2 = model(frame2)

        # frame = result1[0].plot()
        # frame2 = result2[0].plot()

        # Display the annotated frame
        # out.write(frame)
        # out2.write(frame)

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