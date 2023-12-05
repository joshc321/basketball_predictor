"""
Calculates the overlap distance of two cameras 
with the same FOV on the same plane
given there distance from eachother and distance to 
measuring plane

UNITS : units being used for calculations
CAM_GAP : distance between centers of cameras
CAM_FOV : FOV of the cameras in degress
VIEW_DIST : distance from camera plane to measuring plane for overlap
"""


import math


UNITS = "in"
CAM_GAP = 30.75
CAM_FOV = 55
VIEW_DIST = 151

def calc_overlap(cam_gap, fov, dist):
    """
    """

    fov = math.radians(fov)

    return (dist * 2 * math.sin(fov / 2) / math.sin( (math.pi - fov) / 2 )) - cam_gap

if __name__ == '__main__':

    overlap_amt = calc_overlap(CAM_GAP, CAM_FOV, VIEW_DIST)

    print(f'CAM GAP: {CAM_GAP} {UNITS}\nFOV: {CAM_FOV} degrees\nVIEW DIST: {VIEW_DIST} {UNITS}\nOverlap: {overlap_amt} {UNITS}')