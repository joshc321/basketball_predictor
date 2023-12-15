"""
Script to run YOLO
object detection on video framess
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from ball_fun.sterio_cameras import SterioCameras


cams = SterioCameras(["./media/demo2_left.mp4", "./media/demo2_right.mp4"])
cams.show_classification_stream(YOLO(f'{Path(__file__).parent.parent.parent}/YOLO_model/models/best.pt'))
cams.close()
