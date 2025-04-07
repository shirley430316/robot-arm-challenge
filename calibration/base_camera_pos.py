import gripper_base_pos
import apriltag_pos
import cv2
from pydobotplus import dobotplus as dobot
import time
import numpy as np

'''
In this file, a_b means aHb, the transformation matrix from b to a.

Usage:
    import base_camera_pos
    base_camera, gripper_tag = base_camera_pos.get_base_camera_pos(independent=True)
'''


def get_base_camera_pos(independent=True):
    '''
    Turn independent=True if you want to run this function independently, otherwise set it to False.
    If independent=False, the function will not close the camera and Dobot connection, after the function is done.
    '''

    # Initialize Dobot
    device = dobot.Dobot(port='COM3')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Adjust camera index as needed
    
    # Define calibration positions (x,y,z,r in mm and degrees)
    calibration_positions = [
        [270, 0, 40,0],
        [265, -2, 40, 0],  # Position 2
        [260, -4, 40, 0],    # Position 3
        [260, 4, 50, 0],      # Position 4
        [265, 2, 50, 0],      # Position 5
        [270, 0, 50, 0],      # Position 6
        [275, 0, 50, 0],      # Position 7
        [275, 0, 40, 0],      # Position 8
        ]    # Data collection containers
    tag_camera_poses = []
    gripper_base_poses = []
    
    try: 
        for i, pos in enumerate(calibration_positions):
            print(f"Moving to position {i+1}/{len(calibration_positions)}: {pos}")
            
            # Move to position

            device.move_to(*pos)
            time.sleep(2)  # Wait for movement to settle

            # Capture AprilTag pose
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue
                
            tag_camera_pose = apriltag_pos.find_apriltag_transform()["transform_matrix"]
            tag_camera_poses.append(tag_camera_pose)
            print("Captured AprilTag pose: ")
            print(tag_camera_pose)
            
            # Get current gripper pose (convert to meters)
            gripper_base_pose = gripper_base_pos.get_gripper_transform_matrix(device=device) 
            gripper_base_poses.append(gripper_base_pose)
            print("Captured gripper_base pose: ")
            print(gripper_base_pose)

            
            print(f"Collected data point {i+1}")
            
        # Calibrate transformations
        print("Calculating transformations...")
        base_camera, gripper_tag = calibrate_transforms(tag_camera_poses, gripper_base_poses)
        
        print("\nCalibration results:")
        print("\ngripper_tag:\n", gripper_tag)
        print("\nbase_camera:\n", base_camera)
        print("\ncamera_base:\n", np.linalg.inv(base_camera))
    
    finally:
        if independent:
            device.close()
            cap.release()
            np.save("base_camera_result.npy", base_camera)

def calibrate_transforms(A_list, B_list):
    """
    Solves for X (base_camera) and Y (gripper_tag) in the equation: X = B⁻¹ @ Y @ A
    
    Args:
        A_list: List of tag_camera poses (4x4 matrices)
        B_list: List of gripper_base poses (4x4 matrices)
    
    Returns:
        X (base_camera), Y (gripper_tag) as 4x4 homogeneous matrices
    """
    # Ensure we have at least 3 poses (minimum for 6DoF calibration)
    assert len(A_list) >= 3, "Need ≥3 poses for unique solution"
    
    # Initialize system matrices
    M = np.zeros((12 * len(A_list), 24))  # 12 equations per pose pair (6 rot + 6 trans)
    b = np.zeros((12 * len(A_list), 1))
    
    for i, (A, B) in enumerate(zip(A_list, B_list)):
        # Extract rotation (R) and translation (t) components
        RA = A[:3, :3]
        tA = A[:3, 3].reshape(-1, 1)
        RB = B[:3, :3]
        tB = B[:3, 3].reshape(-1, 1)
        
        # --- Rotation part: RX = RB⁻¹ @ RY @ RA ---
        # Equivalent to: RB @ RX = RY @ RA
        # Vectorized as: (RA.T ⊗ RB) vec(RX) - (I ⊗ I) vec(RY) = 0
        M[12*i : 12*i+9, 0:9] = np.kron(RA.T, RB)  # Coefficients for RX
        M[12*i : 12*i+9, 9:18] = -np.kron(np.eye(3), np.eye(3))  # Coefficients for RY
        
        # --- Translation part: tX = RB⁻¹ @ (RY @ tA + tY - tB) ---
        # Rewrite as: RB @ tX - RY @ tA - tY = -tB
        M[12*i+9 : 12*i+12, 0:3] = RB  # Coefficients for tX
        M[12*i+9 : 12*i+12, 9:18] = -np.kron(tA.T, np.eye(3))  # Coefficients for RY
        M[12*i+9 : 12*i+12, 18:21] = -np.eye(3)  # Coefficients for tY
        b[12*i+9 : 12*i+12] = -tB
    
    # Solve least-squares problem
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    
    # Extract RX (3x3), tX (3x1), RY (3x3), tY (3x1)
    RX = x[0:9].reshape(3, 3)
    RY = x[9:18].reshape(3, 3)
    tX = x[18:21].reshape(3, 1)
    tY = x[21:24].reshape(3, 1)
    
    # Orthogonalize rotations (nearest SO(3))
    UX, _, VX = np.linalg.svd(RX)
    RX = UX @ VX
    UY, _, VY = np.linalg.svd(RY)
    RY = UY @ VY
    
    # Build 4x4 homogeneous matrices
    X = np.eye(4)
    X[:3, :3] = RX
    X[:3, 3] = tX.flatten()
    
    Y = np.eye(4)
    Y[:3, :3] = RY
    Y[:3, 3] = tY.flatten()
    
    return X, Y

if __name__ == "__main__":
    get_base_camera_pos(independent=True)