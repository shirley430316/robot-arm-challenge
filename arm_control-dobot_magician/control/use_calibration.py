import sys
import os
import numpy as np

# Add the calibration directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "calibration"))

# Import calibration modules
import gripper_base_pos
import apriltag_pos
import gripper_tag_pos
import base_camera_pos_v2

# Import robot control modules
from mydobot import MyDobot

def main():
    """
    Example of using the calibration files with the robot arm.
    This script demonstrates how to:
    1. Load a previously saved calibration result
    2. Use the calibration to transform coordinates between camera and robot frames
    """
    try:
        # Initialize the robot
        device = MyDobot()
        
        # Load the calibration result if it exists
        calibration_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                       "calibration", "base_camera_result.npy")
        
        if os.path.exists(calibration_path):
            print(f"Loading calibration from: {calibration_path}")
            base_camera = np.load(calibration_path)
            print("Loaded base_camera transformation matrix:")
            print(base_camera)
        else:
            print("No calibration file found. Please run the calibration first.")
            print("You can run: python base_camera_pos-v2.py from the calibration directory")
            return
        
        # Example: Get the current gripper pose
        gripper_base = gripper_base_pos.get_gripper_transform_matrix(device=device)
        print("\nCurrent gripper pose (relative to robot base):")
        print(gripper_base)
        
        # Example: Get the gripper to tag transformation
        gripper_tag = gripper_tag_pos.get_gripper_tag_transform_matrix()
        print("\nGripper to tag transformation:")
        print(gripper_tag)
        
        # Example: Calculate the camera to base transformation
        # This is the same as the loaded base_camera matrix
        print("\nCamera to base transformation (from loaded calibration):")
        print(base_camera)
        
        # Example: Move the robot to a safe position
        print("\nMoving robot to a safe position...")
        device.move_to(200, 0, 50, 0)
        
        print("\nCalibration example completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the robot connection
        device.close()

if __name__ == "__main__":
    main() 