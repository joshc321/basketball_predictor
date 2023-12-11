import numpy as np
from matplotlib import pyplot as plt
import time

def fit_trajectory(points: np.ndarray):
    """
    Fits a trajectory to a set of 3D points.

    Args:
        points (np.ndarray): The 3D points to fit the trajectory to of shape (3,N).

    Returns:
        (np.ndarray): The coefficients of the fitted trajectory
    """

    x, y, z = points

    xy_fit, resxy, _,_,_ = np.polyfit(x,y,2, full=True)
    xz_fit, resxz, _,_,_ = np.polyfit(x,z,2, full=True)

    return np.poly1d(xy_fit), np.poly1d(xz_fit), resxz + resxy

def predict_future_positions(points: np.ndarray, xy_fit, xz_fit, num_steps):
    """
    Predicts the future positions of a basketball based on its trajectory.

    Args:
        points (np.ndarray): The known 3D coordinates of the basketball along its trajectory path.
        params (np.ndarray): The coefficients of the fitted trajectory.
        num_steps (int): The number of future positions to predict.

    Returns:
        np.ndarray: The predicted future positions of the basketball.
    """

    # Get the current position of the basketball
    x_vals = np.linspace(points[0,0], points[0,-1] + 1000, num_steps)

    # Predict the future positions
    future_positions = np.zeros((3, num_steps))
    for i in range(num_steps):
        x = x_vals[i]
        y = xy_fit(x)
        z = xz_fit(x)

        future_positions[:,i] = np.array([x, y, z]).T

    return future_positions


# Collect a set of 3D points
# points = np.array([[2.9217, 2, 3], [2, 3, 4], [3, 4, 5]])
points = np.array([
    [292.17,484.64,775.81,1055.8],
    [338.28,423.92,480.82,451.5],
    [3685.4,3703.5,3694.6,3651.9]
])

# Fit a trajectory to the points
s = time.time_ns()
xy_fit, xz_fit, res = fit_trajectory(points)

# Predict the future positions of the basketball
num_steps = 100
future_positions = predict_future_positions(points, xy_fit, xz_fit, num_steps)

print(time.time_ns() - s)
print('error', res)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[0,:], points[1,:], points[2,:], 'bo')
ax.plot(future_positions[0,:],future_positions[1,:],future_positions[2,:], color='r')
plt.show()