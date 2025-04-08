import numpy as np
from math import cos, sin, radians
from pydobotplus import dobotplus as dobot

def get_gripper_transform_matrix(device=None, port='COM7', independent=False):

    if device is None:
        # Initialize Dobot (now correctly calling the class)
        robot = dobot.Dobot(port="COM7")
    else:
        robot = device

    # Get current pose (x, y, z, r)
    pose = robot.get_pose()  # Returns (x, y, z, r) in mm and degrees
    print(f"Pose: {pose}")
    x, y, z, r_deg = pose[0]
    
    if independent:
        # Close connection
        robot.close()

    # Convert rotation angle (degrees → radians)
    r = np.arctan(y/x)

    # Construct rotation matrix (Z-axis only)
    R = np.array([
        [cos(r), sin(r), 0],
        [-sin(r),  cos(r), 0],
        [0,      0,      1]
    ])

    # Translation vector (3x1)
    T = np.array([[x], [y], [z]])

    # Homogeneous transformation matrix [[R T], [0 1]]
    H = np.vstack([
        np.hstack([R, T]),          # [[R T]
        np.array([0, 0, 0, 1])      #  [0 1]]
    ])

    return H

if __name__ == "__main__":
    H = get_gripper_transform_matrix(port='COM7')  # Replace with your port
    print("Homogeneous Transformation Matrix (Gripper to Base):")
    print(np.round(H, 3))  # Round for readability
