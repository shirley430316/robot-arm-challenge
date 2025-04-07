import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
from pathlib import Path
import time
from threading import Event

# Global variables for camera and model
camera = None
model = None
pipeline = None
align = None

# Lists to store detection results for each class
big_objects = []
box_objects = []
circle_objects = []
small_objects = []

def initialize_camera():
    """
    Initialize the RealSense camera and return True if successful
    """
    global camera, pipeline, align
    
    try:
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline
        pipeline_profile = pipeline.start(config)
        
        # Create align object to align depth to color
        align = rs.align(rs.stream.color)
        
        # Wait for camera to stabilize
        for _ in range(30):
            pipeline.wait_for_frames()
            
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return False

def initialize_model(model_path="yolo_ws/model/model.pt"):
    """
    Initialize the YOLO model for object detection.
    
    Args:
        model_path: Path to the YOLO model weights file
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global model
    
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def capture_frame():
    """
    Capture a frame from the camera and return color and depth images
    """
    global pipeline, align
    
    if pipeline is None:
        print("Camera not initialized. Call initialize_camera() first.")
        return None, None
    
    try:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Align depth to color
        aligned_frames = align.process(frames)
        
        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Failed to get frames")
            return None, None
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None, None

def detect_objects(color_frame, depth_frame, conf_threshold=0.15, iou_threshold=0.1, max_det=50, pipeline=None):
    """
    Detect objects in the given frame using YOLO model
    Returns a list of detections and additional coordinate information
    """
    global model
    
    if model is None:
        print("Model not initialized. Call initialize_model() first.")
        return None, None, None
    
    # Convert frame to RGB for YOLO
    frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    
    # Get frame dimensions
    height, width = depth_frame.shape[:2]
    
    # Run YOLO detection
    results = model.predict(
        source=frame_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False
    )[0]
    
    detections = []
    best_board = None
    top_boxes = []
    
    if hasattr(results, 'obb') and results.obb is not None:
        # Process all detections
        for i in range(len(results.obb.cls)):
            class_id = int(results.obb.cls[i])
            class_name = results.names[class_id]
            confidence = float(results.obb.conf[i])
            
            # Get corner points
            points = results.obb.xyxyxyxy[i].cpu().numpy()
            box_points = points.reshape((-1, 2))
            
            # Calculate center point
            center_x = int(np.mean(points[::2]))
            center_y = int(np.mean(points[1::2]))
            
            # Ensure center points are within image bounds
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))
            
            # Get depth at center point
            depth = depth_frame[center_y, center_x]
            
            # Get angle
            angle = float(results.obb.xywhr[i][-1])
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'box_points': box_points,
                'position_2d': (center_x, center_y),
                'depth': depth,
                'angle': angle
            }
            
            detections.append(detection)
            
            # Track best board
            if class_name == 'board':
                if best_board is None or confidence > best_board['confidence']:
                    best_board = detection
            
            # Track top boxes
            if class_name == 'box':
                top_boxes.append(detection)
    
    # Sort boxes by confidence and get top 3
    if top_boxes:
        top_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        top_boxes = top_boxes[:3]
    
    # Get 3D coordinates for top boxes
    box_coords_3d = []
    for box in top_boxes:
        position_2d_angle = (box['position_2d'][0], box['position_2d'][1], box['angle'])
        coords_3d = get_3d_coordinates(position_2d_angle, depth=box['depth'], pipeline=pipeline)
        if coords_3d:
            box_coords_3d.append(coords_3d[:3])  # Only take x, y, z coordinates
    
    # Get board corner coordinates if found
    board_corners = None
    if best_board:
        board_corners = best_board['box_points'].tolist()
    
    return detections, board_corners, box_coords_3d

def get_best_detection(class_name):
    """
    Return the detection with highest confidence for the specified class
    Only supports 'big', 'small', and 'circle' classes
    """
    if class_name == "big" and big_objects:
        return max(big_objects, key=lambda x: x["confidence"])
    elif class_name == "circle" and circle_objects:
        return max(circle_objects, key=lambda x: x["confidence"])
    elif class_name == "small" and small_objects:
        return max(small_objects, key=lambda x: x["confidence"])
    else:
        return None

def get_3d_coordinates(position_2d_angle, depth=None, pipeline=None):
    """
    Convert 2D position and angle to 3D coordinates in camera frame
    If depth is provided, use it directly; otherwise, get depth from camera
    """
    if pipeline is None:
        print("Camera pipeline not provided")
        return None
    
    try:
        # Extract position and angle
        x, y = position_2d_angle[:2]
        angle = position_2d_angle[2] if len(position_2d_angle) > 2 else 0
        
        # Convert depth to meters
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        depth_meters = depth * depth_scale if depth is not None else None
        
        # Get camera intrinsics
        color_profile = pipeline.get_active_profile().get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Convert to 3D coordinates
        x_3d = (x - intrinsics.ppx) / intrinsics.fx * depth_meters
        y_3d = (y - intrinsics.ppy) / intrinsics.fy * depth_meters
        z_3d = depth_meters
        
        return (x_3d, y_3d, z_3d, angle)
    except Exception as e:
        print(f"Error getting 3D coordinates: {e}")
        return None

def close_camera():
    """
    Close the camera connection
    """
    global pipeline
    
    if pipeline:
        pipeline.stop()
        print("Camera closed")

def realtime_detect(pipeline, stop_event=None, detection_results=None):
    """
    Real-time object detection using RealSense camera.
    
    Args:
        pipeline: RealSense pipeline object
        stop_event: Event to signal when to stop (optional)
        detection_results: Dictionary to store detection results (optional)
        
    Returns:
        None
    """
    if stop_event is None:
        stop_event = Event()
    if detection_results is None:
        detection_results = {}
        
    try:
        while not stop_event.is_set():
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect objects
            detections, board_corners, box_coords_3d = detect_objects(
                color_image,
                depth_image,
                conf_threshold=0.15,
                iou_threshold=0.1,
                max_det=50,
                pipeline=pipeline
            )
            
            if detections:
                # Store the detection results
                detection_results['detections'] = detections
                detection_results['board_corners'] = board_corners
                detection_results['box_coords_3d'] = box_coords_3d
                detection_results['timestamp'] = time.time()
                
                # Find the best detection for each class
                best_detections = {'board': None, 'circle': None, 'small': None, 'big': None}
                
                for detection in detections:
                    class_name = detection['class']
                    if class_name in best_detections:
                        if (best_detections[class_name] is None or 
                            detection['confidence'] > best_detections[class_name]['confidence']):
                            best_detections[class_name] = detection
                
                # Store the best detections
                detection_results['best_detections'] = best_detections
                
                # Get 3D coordinates for the best detections
                for class_name, detection in best_detections.items():
                    if detection:
                        position_2d_angle = (
                            detection['position_2d'][0],
                            detection['position_2d'][1],
                            detection['angle']
                        )
                        coords_3d = get_3d_coordinates(position_2d_angle, depth=detection['depth'], pipeline=pipeline)
                        if coords_3d:
                            detection_results[f'{class_name}_coords_3d'] = coords_3d
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Detection error: {e}") 

def is_point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: (x, y) coordinates of the point to check
        polygon: List of (x, y) coordinates defining the polygon vertices
        
    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside 

def find_best_objects(color_image, depth_image, pipeline=None):
    """
    Find the best objects in the image:
    1. Best board
    2. Best small, big, circle that are inside the board
    3. Best 3 boxes
    
    Args:
        color_image: Color image from camera
        depth_image: Depth image from camera
        pipeline: RealSense pipeline (optional)
        
    Returns:
        tuple: (best_board, best_inside_board, best_boxes, coords_3d_dict, visualization_image)
    """
    # Detect objects
    detections, board_corners, box_coords_3d = detect_objects(
        color_image,
        depth_image,
        conf_threshold=0.15,
        iou_threshold=0.1,
        max_det=50,
        pipeline=pipeline
    )
    
    if not detections:
        return None, None, None, None, color_image.copy()
    
    # Create visualization image
    visualization_image = color_image.copy()
    
    # Find the best board
    best_board = None
    for detection in detections:
        if detection['class'] == 'board':
            if best_board is None or detection['confidence'] > best_board['confidence']:
                best_board = detection
    
    # Find objects inside the board
    best_inside_board = {'circle': None, 'small': None, 'big': None}
    if best_board and board_corners:
        board_polygon = board_corners
        
        # Find the best objects inside the board
        for detection in detections:
            class_name = detection['class']
            if class_name in ['circle', 'small', 'big']:
                # Check if the object is inside the board
                center_point = detection['position_2d']
                if is_point_inside_polygon(center_point, board_polygon):
                    if (best_inside_board[class_name] is None or 
                        detection['confidence'] > best_inside_board[class_name]['confidence']):
                        best_inside_board[class_name] = detection
    
    # Find the best 3 boxes
    box_detections = []
    for detection in detections:
        if detection['class'] == 'box':
            box_detections.append(detection)
    
    # Sort boxes by confidence and get top 3
    box_detections.sort(key=lambda x: x['confidence'], reverse=True)
    best_boxes = box_detections[:3]
    
    # Get 3D coordinates for all objects
    coords_3d_dict = {}
    
    # Get 3D coordinates for objects inside the board
    for class_name, detection in best_inside_board.items():
        if detection:
            position_2d_angle = (
                detection['position_2d'][0],
                detection['position_2d'][1],
                detection['angle']
            )
            coords_3d = get_3d_coordinates(position_2d_angle, depth=detection['depth'], pipeline=pipeline)
            if coords_3d:
                coords_3d_dict[class_name] = coords_3d
    
    # Get 3D coordinates for boxes
    for i, box in enumerate(best_boxes):
        position_2d_angle = (
            box['position_2d'][0],
            box['position_2d'][1],
            box['angle']
        )
        coords_3d = get_3d_coordinates(position_2d_angle, depth=box['depth'], pipeline=pipeline)
        if coords_3d:
            coords_3d_dict[f'box_{i+1}'] = coords_3d
    
    # Draw the objects on the visualization image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    # Draw the board
    if best_board:
        points = best_board['box_points'].astype(np.int32)
        cv2.polylines(visualization_image, [points], True, (0, 0, 255), thickness)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        cv2.putText(visualization_image, "Board", (center_x, center_y), font, font_scale, (0, 0, 255), thickness)
    
    # Draw objects inside the board
    colors = {
        'circle': (0, 255, 0),  # Green
        'small': (0, 255, 255),  # Yellow
        'big': (255, 255, 0)    # Cyan
    }
    
    for class_name, detection in best_inside_board.items():
        if detection:
            points = detection['box_points'].astype(np.int32)
            color = colors[class_name]
            cv2.polylines(visualization_image, [points], True, color, thickness)
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            cv2.putText(visualization_image, class_name.capitalize(), (center_x, center_y), font, font_scale, color, thickness)
    
    # Draw boxes
    for i, box in enumerate(best_boxes):
        points = box['box_points'].astype(np.int32)
        cv2.polylines(visualization_image, [points], True, (255, 0, 0), thickness)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        cv2.putText(visualization_image, f"Box {i+1}", (center_x, center_y), font, font_scale, (255, 0, 0), thickness)
    
    return best_board, best_inside_board, best_boxes, coords_3d_dict, visualization_image 