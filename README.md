# basketball_predictor

Program to detect if a thrown ball will go into 
the hoop.

Steps:
- sterio camera calibration
- ball detection
- ball tracking
- hoop detection
- triangulation
- shot prediction

Demo video: https://youtu.be/pWDL0jkxISw



### Sterio Cameras
The cameras used were a pair of Logitech C270 webcams taped to a metal pole. These were then calibrated using functions from openCV.
Calibration of your own cameras can be done by running the file [camera_calibration.py](./tools/camera_calibration/camera_calibration.py) and modifying it
to match the steps needed by your camera.

### Ball Detection
Detection was achieved through a hybrid approch using the YOLOv8 image classification model to initially get the ball parameters needed for hough circles.
After the initial calibration hough circles are used along with background subtraction as the main filter.

### Ball Tracking
The tracking algorithm developed relies on creating a graph structure of the balls locations in 3D space. Each new location is added to the graph and connected to all possible previous points.

### Hoop detection
Hoop detection is done through the YOLOv8 model trained on a custom dataset.

### Triangulation
Traingulation is done using direct linear transform. More on this can be found [here](https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html)

### Shot Prediction
Prediction is done by fitting a pair of 2D trajectories to the XY values and the XZ values of each possible path determined by the tracking algorithm. The best path is then chosen and can be used to predict if the ball will go into the hoop.
