"""
DA FILE to run da predictor thing or sums
"""

from ball_fun.object_tracking import tracking_stream
from ball_fun.sterio_cameras import SterioCameras


if __name__ == '__main__':

    # sterio_pair = SterioCameras(["./media/test_vid_104_L.mp4", "./media/test_vid_105_R.mp4"])
    sterio_pair = SterioCameras()

    tracking_stream(sterio_pair)
    # sterio_pair.show_stream(width_scale=0.5, height_scale=0.5)
    sterio_pair.close()

