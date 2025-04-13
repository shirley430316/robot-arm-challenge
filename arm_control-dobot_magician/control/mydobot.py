import time
import sys
import os
import numpy as np
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pydobotplus'))

from pydobotplus import dobotplus as dobot
from pydobotplus.dobotplus import MODE_PTP  # Import MODE_PTP directly

class MyDobot(dobot.Dobot):
    def __init__(self, port=None, config_path=None):
        """
        Initialize the MyDobot class.
        
        Args:
            port: Optional port name (e.g., 'COM5'). If None, it will try to get it from config.
            config_path: Optional path to the config file. If None, it will use the default path.
        """
        super().__init__(port=port, config_path=config_path)
        
    def jump_to(self, x, y, z, r, height=50, wait=False):
        """Move to a position with a jump motion."""
        current_pose = self.get_pose().position
        self._set_ptp_cmd(current_pose.x, current_pose.y, current_pose.z + height, current_pose.r, mode=MODE_PTP.MOVL_XYZ, wait=True)
        self._set_ptp_cmd(x, y, z + height, r, mode=MODE_PTP.MOVL_XYZ, wait=True)
        self._set_ptp_cmd(x, y, z, r, mode=MODE_PTP.MOVL_XYZ, wait=wait)

    def find_nearest_point_on_curve(self, point, curve):
        """
        Find the nearest point on a curve to a given point.
        
        Args:
            point: [x, y, z, r] coordinates of the point
            curve: List of [x, y, z, r] coordinates representing points on a curve
            
        Returns:
            [x, y, z, r] coordinates of the nearest point on the curve
        """
        # Convert to numpy arrays for easier calculation
        point = np.array(point)
        curve = np.array(curve)
        
        # Calculate distances to all points on the curve
        # We only consider x, y, z for distance calculation, not r
        distances = np.sqrt(np.sum((curve[:, :3] - point[:3])**2, axis=1))
        
        # Find the index of the nearest point
        nearest_idx = np.argmin(distances)
        
        # Return the nearest point
        return curve[nearest_idx].tolist()

    def safe_jump_to(self, safe_curve=[[82.72747802734375, 239.459716796875, 38.95375061035156, 79.07131958007812]], destination=[], wait=True):
        """
        Safely move the robot arm to a destination using a predefined safe curve.
        
        Args:
            safe_curve: List of [x, y, z, r] coordinates representing points on a safe curve
                       where the arm can move safely by only rotating the first joint
            destination: [x, y, z, r] coordinates of the destination point
            wait: Whether to wait for movement completion
            
        Returns:
            bool: True if movement was successful, False otherwise
        """
        try:
            # Get current position
            current_pose = self.get_pose().position
            current_pos = [current_pose.x, current_pose.y, current_pose.z, current_pose.r]
            
            # Find the nearest point on the safe curve to the current position
            nearest_safe_point = self.find_nearest_point_on_curve(current_pos, safe_curve)
            
            # Find the nearest point on the safe curve to the destination
            nearest_safe_to_dest = self.find_nearest_point_on_curve(destination, safe_curve)
            
            # Step 1: Move to the nearest point on the safe curve (lifting first)
            self._set_ptp_cmd(nearest_safe_point[0], nearest_safe_point[1], nearest_safe_point[2], 
                         nearest_safe_point[3], mode=MODE_PTP.MOVJ_XYZ, wait=True)
            
            # Step 2: Move along the safe curve by rotating the first joint
            self._set_ptp_cmd(nearest_safe_to_dest[0], nearest_safe_to_dest[1], nearest_safe_to_dest[2], 
                         nearest_safe_to_dest[3], mode=MODE_PTP.MOVJ_XYZ, wait=True)
            
            # Step 3: Move to the final destination
            self._set_ptp_cmd(destination[0], destination[1], destination[2], 
                         destination[3], mode=MODE_PTP.MOVJ_XYZ, wait=wait)
            
            return True
            
        except Exception as e:
            print(f"Safe movement failed with error: {str(e)}")
            return False

    def move_joint(self, joint_idx, angle, wait=True):
        """
        Move a single joint to a specific angle.
        
        Args:
            joint_idx: Index of the joint to move (0-3)
            angle: Target angle in degrees
            wait: Whether to wait for movement completion
            
        Returns:
            bool: True if movement was successful, False otherwise
        """
        try:
            # Get current joint angles
            current_joints = self.get_pose().joints
            joints = [current_joints.j1, current_joints.j2, current_joints.j3, current_joints.j4]
            
            # Update the specified joint
            joints[joint_idx] = angle
            
            # Move to the new joint configuration
            self._set_ptp_cmd(joints[0], joints[1], joints[2], joints[3], 
                          MODE_PTP.MOVJ_ANGLE, wait=wait)
            
            return True
        except Exception as e:
            print(f"Joint movement failed with error: {str(e)}")
            return False

    def safe_move_to(self, x=None, y=None, z=None, r=0, wait=True, max_retries=3):
        """
        Safely move the robot arm to a position with error handling.
        
        Args:
            x, y, z, r: Target coordinates
            wait: Whether to wait for movement completion
            max_retries: Maximum number of retry attempts if movement fails
            
        Returns:
            bool: True if movement was successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Check for any existing alarms
                alarms = self.get_alarms()
                if alarms:
                    print(f"Clearing alarms before movement: {alarms}")
                    self.clear_alarms()
                    time.sleep(1)  # Wait for alarms to clear
                    
                # Perform the movement
                self.move_to(x, y, z, r, wait)
                
                # Check for new alarms after movement
                alarms = self.get_alarms()
                if not alarms:
                    return True
                else:
                    print(f"Movement failed with alarms: {alarms}")
                    self.clear_alarms()
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Movement attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
                continue
                
        return False

    def suck(self, enable, wait_time=0):
        """Control the suction cup."""
        self._set_end_effector_suction_cup(enable)
        time.sleep(wait_time)    
        
    def grip(self, enable, wait_time=0):
        """Control the gripper."""
        self._set_end_effector_gripper(enable)
        time.sleep(wait_time)

