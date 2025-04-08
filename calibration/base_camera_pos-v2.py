import gripper_base_pos
import apriltag_pos
import gripper_tag_pos
import cv2
from pydobotplus import dobotplus as dobot
import time
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

'''
In this file, a_b means aHb, the transformation matrix from b to a.

Usage:
    import base_camera_pos
    base_camera, gripper_tag = base_camera_pos.get_base_camera_pos(independent=True)
'''

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Higher resolution for better detection

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_params = [intr.fx, intr.fy, intr.ppx, intr.ppy]

# Create 3x3 camera matrix for OpenCV functions
camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])

print(f"Camera parameters: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")


def get_base_camera_pos(independent=True):
    '''
    Turn independent=True if you want to run this function independently, otherwise set it to False.
    If independent=False, the function will not close the camera and Dobot connection, after the function is done.
    '''

    base_camera = None

    # Initialize Dobot
    device = dobot.Dobot(port='COM7')
    
    
    # Define calibration positions (x,y,z,r in mm and degrees)
    calibration_positions = [
        [270, 0, 20,0],
        [150, -150, 20, 0],  # Position 2
        [200, -50, 20, 0],    # Position 3
        [260, 4, 30, 0],      # Position 4
        [220, -30, 30, 0],      # Position 6
        [200, 0, 30, 0],      # Position 7
        [223, 0, 10, 0],      # Position 8
        ]    # Data collection containers
    tag_camera_poses = []
    gripper_base_poses = []
    sample_base_camera_poses = []

    gripper_tag = gripper_tag_pos.get_gripper_tag_transform_matrix()
    
    try: 
        for i, pos in enumerate(calibration_positions):
            print(f"Moving to position {i+1}/{len(calibration_positions)}: {pos}")
            
            # Move to position

            device.move_to(*pos)
            time.sleep(2)  # Wait for movement to settle

                
            tag_camera_pose = apriltag_pos.find_apriltag_transform(pipeline=pipeline, camera_matrix=camera_matrix, camera_params=camera_params)["transform_matrix"]
            tag_camera_poses.append(tag_camera_pose)
            
            # Get current gripper pose (convert to meters)
            gripper_base_pose = gripper_base_pos.get_gripper_transform_matrix(device=device) 
            gripper_base_poses.append(gripper_base_pose)
            print("Captured gripper_base pose: ")
            print(gripper_base_pose)

            sample_base_camera = gripper_base_pose @ gripper_tag @ tag_camera_pose
            sample_base_camera_poses.append(sample_base_camera)
            

            print(f"Collected data point {i+1}")
            print("sample_base_camera:\n", sample_base_camera)

        print("Calculating average transformation...")
        # base_camera
        rotation_matrix = extract_and_average_rotation(sample_base_camera_poses)
        base_camera_vinilla_mean = np.mean(sample_base_camera_poses, axis=0)
        base_camera_vinilla_mean[:3, :3] = rotation_matrix

        base_camera = base_camera_vinilla_mean


    finally:
        if independent:
            device.close()
            pipeline.stop()
            if base_camera is not None:
                print("base_camera:\n", base_camera)
                np.save("base_camera_result.npy", base_camera)
            else:
                print("No base_camera result to save.")

def extract_and_average_rotation(matrices):
    euler_angles = []
    
    for matrix in matrices:
        # Extract 3x3 rotation matrix
        rotation_matrix = matrix[:3, :3]
        
        # Convert to Euler angles (ZYX convention)
        euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
        euler_angles.append(euler_zyx)
    
    # Average Euler angles (handling wraparound)
    euler_angles = np.array(euler_angles)
    avg_euler = np.arctan2(np.mean(np.sin(euler_angles), axis=0), 
                           np.mean(np.cos(euler_angles), axis=0))
    
    # Convert averaged Euler angles back to rotation matrix
    avg_rotation = R.from_euler('zyx', avg_euler).as_matrix()
    
    return avg_rotation

get_base_camera_pos()