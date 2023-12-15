from dataclasses import dataclass

@dataclass
class Parameters:

    """
    Ball detector params
    """

    # if using ball_detector.HoughBallDetector
    minRadius: int = 25
    maxRadius: int = 50

    # hough circle params
    dp: float = 1
    minDist: float = 300
    param1: float = 300
    param2: float = 12


    """
    tracking params
    """
    max_time: int = 1
    min_speed: float = 0
    max_speed: float = 20