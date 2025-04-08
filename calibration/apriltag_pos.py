import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector

'''
Usage:
    import apriltag_pos
    tag_camera = apriltag_pos.find_apriltag_transform()['transform_matrix']	

return:
    {"
    transform_matrix": transform matrix from camera to apriltag, 
    "tag_id": the id of the detected AprilTag
    }
'''

def create_4x4_transform_matrix(rotation, translation):
    """Create a 4x4 transformation matrix from rotation and translation"""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.flatten()
    return transform

# Configure AprilTag detector
at_detector = Detector(
    families='tag36h11',
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25
)

TAG_SIZE = 0.1  # Size of the AprilTag in meters
''''
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
'''


from collections import deque

def create_4x4_transform_matrix(R, t):
    """Create a 4x4 transformation matrix from rotation matrix and translation vector"""
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t.flatten()
    return transform

def is_stabilized(transform_history, threshold=0.001, window_size=10):
    """Check if the transformation has stabilized"""
    if len(transform_history) < window_size:
        return False
    
    # Calculate differences between consecutive matrices
    diffs = []
    for i in range(1, len(transform_history)):
        diff = np.linalg.norm(transform_history[i] - transform_history[i-1])
        diffs.append(diff)
    
    # Check if all recent differences are below threshold
    return all(d < threshold for d in diffs[-window_size:])

def find_apriltag_transform(independent=False, pipeline=None, camera_matrix=None, camera_params=None):
    "if calling this function outside this file, set independent=False, otherwise pipeline would be stopped"
    try:
        # Parameters
        STABILITY_THRESHOLD = 1  # Threshold for considering the tag stable
        STABILITY_WINDOW = 15        # Number of frames to check for stability
        transform_history = deque(maxlen=STABILITY_WINDOW)
        stable_transform = None
        tag_id = None
        
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags with camera parameters
            tags = at_detector.detect(
                gray_image, 
                estimate_tag_pose=True, 
                camera_params=camera_params, 
                tag_size=TAG_SIZE
            )
            
            current_transform = None
            current_tag_id = None
            
            for tag in tags:
                if tag.pose_R is not None and tag.pose_t is not None:
                    current_tag_id = tag.tag_id
                    current_transform = create_4x4_transform_matrix(tag.pose_R.T, - tag.pose_R.T @ tag.pose_t * 1000)  # Convert to mm
                    
                    # Visualization
                    for idx in range(len(tag.corners)):
                        cv2.line(color_image, tuple(tag.corners[idx-1].astype(int)), 
                                 tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
                    
                    cv2.putText(color_image, f"ID: {tag.tag_id}",
                               (int(tag.center[0]), int(tag.center[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Draw coordinate axes
                    axis_length = 0.1  # 10cm length for axes
                    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
                    
                    # Project 3D points to 2D image plane
                    imgpts, _ = cv2.projectPoints(axis_points, 
                                                 tag.pose_R, tag.pose_t,
                                                 camera_matrix,
                                                 np.zeros(5))
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    
                    # Draw axes
                    origin = tuple(imgpts[0])
                    cv2.line(color_image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X (red)
                    cv2.line(color_image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y (green)
                    cv2.line(color_image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z (blue)
            
            # Update transform history if we detected a tag
            if current_transform is not None:
                transform_history.append(current_transform)
                tag_id = current_tag_id
                
                # Check if transform has stabilized
                if is_stabilized(transform_history, STABILITY_THRESHOLD, STABILITY_WINDOW):
                    stable_transform = np.mean(transform_history, axis=0)  # Use average of stable period
                    print("\n=== STABLE TRANSFORMATION MATRIX DETECTED ===")
                    print(f"Tag ID: {tag_id}")
                    print("4x4 Transformation Matrix:")
                    print(stable_transform)
                    break  # Exit the loop once stable transform is found

            # Display the image
            cv2.imshow('AprilTag Pose Estimation', color_image)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        if independent:
            pipeline.stop()
            print("Pipeline stopped.")
            cv2.destroyAllWindows()
    
    # Return the stable transformation matrix
    if stable_transform is not None:
        return {
            'transform_matrix': stable_transform,
            'tag_id': tag_id
        }
    else:
        print("No stable transformation detected")
        return None, None
