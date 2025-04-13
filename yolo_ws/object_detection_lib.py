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
        # Check if any RealSense devices are connected
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found. Please check your camera connection.")
            return False, None
        
        print(f"Found {len(devices)} RealSense device(s):")
        for i, device in enumerate(devices):
            print(f"  Device {i+1}: {device.get_info(rs.camera_info.name)}")
            print(f"  Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware version: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams with specific settings
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline with a longer timeout
        profile = pipeline.start(config)
        
        # Get depth sensor and configure it
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # Set depth units to millimeters
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        
        # Enable auto-exposure
        depth_sensor.set_option(rs.option.enable_auto_exposure, True)
        
        # Set visual preset to high accuracy
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset
        
        # Set additional depth sensor options for better accuracy
        depth_sensor.set_option(rs.option.laser_power, 100)  # Maximum laser power
        
        # Create align object
        align = rs.align(rs.stream.color)
        
        # Wait for camera to stabilize and print initial depth frame info
        print("Waiting for camera to stabilize...")
        for i in range(30):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if i % 10 == 0:
                print(f"Frame {i+1}/30 received")
                try:
                    # Try to get a depth value to check if the frame is valid
                    depth_value = depth_frame.get_distance(320, 240)  # Center of the frame
                    print(f"Depth at center: {depth_value} meters")
                    print(f"Depth scale: {depth_sensor.get_depth_scale()}")
                    print(f"Depth units: {depth_sensor.get_option(rs.option.depth_units)}")
                except Exception as e:
                    print(f"Error reading depth at center: {e}")
        
        print("Camera initialized successfully")
        return True, pipeline
        
    except Exception as e:
        print(f"Error initializing camera: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def initialize_model(model_path=None):
    """
    Initialize the YOLO model for object detection.
    
    Args:
        model_path: Path to the YOLO model weights file
        
    Returns:
        tuple: (success, model) where success is a boolean and model is the YOLO model object
    """
    global model
    
    try:
        # If no model path is provided, use the default path
        if model_path is None:
            # Get the absolute path to the model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "model", "mennys_model3.pt")
        
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False, None

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

def detect_objects(image_path):
    """
    Detect objects in an image using YOLO model.
    Uses the same parameters as test_model.py.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of detections, each containing class name and points
    """
    global model
    
    if model is None:
        print("Model not initialized")
        return None
    
    try:
        # Set output directory
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Create a copy for visualization
        img_vis = img.copy()
        
        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run detection with test_model.py parameters
        results = model.predict(
            source=img_rgb,
            conf=0.35,    # Confidence threshold
            iou=0.2,      # IOU threshold
            max_det=30,   # Maximum detections
            verbose=False
        )[0]
        
        detections = []
        
        if hasattr(results, 'obb') and results.obb is not None:
            # Process all detections
            for i in range(len(results.obb.cls)):
                # Get detection info
                class_id = int(results.obb.cls[i])
                class_name = results.names[class_id]
                conf = float(results.obb.conf[i])
                box = results.obb.xyxyxyxy[i].cpu().numpy()
                
                # Get corner points
                points = box.reshape((-1, 2))
                points = points.astype(np.int32)
                
                # Draw rotated box in green
                cv2.polylines(img_vis, [points], True, (0, 255, 0), 2)
                
                # Calculate label position (above the box)
                label_x = int(np.min(points[:, 0]))
                label_y = int(np.min(points[:, 1])) - 5
                
                # Create label text
                label = f"{class_name} ({conf:.2f})"
                
                # Draw label with white background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(img_vis, 
                            (label_x, label_y - text_height - 2),
                            (label_x + text_width, label_y + 2),
                            (255, 255, 255),
                            -1)
                
                # Draw text
                cv2.putText(img_vis, label,
                           (label_x, label_y),
                           font,
                           font_scale,
                           (0, 0, 0),
                           thickness)
                
                # Store detection
                detection = {
                    'class': class_name,
                    'confidence': conf,
                    'points': points
                }
                detections.append(detection)
        
        # Save visualization
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(output_path, img_vis)
        
        return detections
        
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    Returns coordinates in meters
    """
    if pipeline is None:
        print("Camera pipeline not provided")
        return None
    
    try:
        # Extract position and angle
        x, y = position_2d_angle[:2]
        angle = position_2d_angle[2] if len(position_2d_angle) > 2 else 0
        
        # Get camera intrinsics
        color_profile = pipeline.get_active_profile().get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Get depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        # Get depth scale
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print(f"Depth scale: {depth_scale}")
        print(f"Target point: ({x}, {y})")
        
        # Get depth value with validation
        if depth is None:
            # Get depth from a small region around the point to handle noise
            x, y = int(x), int(y)
            depth_values = []
            valid_points = 0
            total_points = 0
            
            # Increase the sampling region
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    total_points += 1
                    try:
                        # Get raw depth value
                        raw_depth = depth_frame.get_distance(x + dx, y + dy)
                        print(f"Raw depth at ({x+dx}, {y+dy}): {raw_depth}")
                        
                        # Convert to meters using depth scale
                        d = raw_depth * depth_scale
                        
                        if 0.1 < d < 2.0:  # Valid depth range (10cm to 2m)
                            depth_values.append(d)
                            valid_points += 1
                    except Exception as e:
                        print(f"Error getting depth at ({x+dx}, {y+dy}): {e}")
                        continue
            
            print(f"Valid depth points: {valid_points}/{total_points}")
            
            if not depth_values:
                print(f"No valid depth values found around point ({x}, {y})")
                return None
                
            # Use median of valid depth values to handle outliers
            depth_meters = np.median(depth_values)
            print(f"Median depth value: {depth_meters}")
        else:
            depth_meters = depth
            print(f"Using provided depth value: {depth_meters}")
            
        if depth_meters is None or depth_meters <= 0 or depth_meters > 2.0:
            print(f"Invalid depth value: {depth_meters}")
            return None
            
        # Convert to 3D coordinates using camera intrinsics
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_meters)
        print(f"3D coordinates: {point_3d}")
        
        # Return the 3D coordinates in meters
        return point_3d
    except Exception as e:
        print(f"Error getting 3D coordinates: {e}")
        import traceback
        traceback.print_exc()
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
    
    print("Starting realtime_detect function...")
    print(f"Pipeline object: {pipeline}")
    
    # Check if pipeline is valid
    if pipeline is None:
        print("Error: Pipeline object is None. Cannot start realtime detection.")
        return
    
    try:
        # Try to get a frame immediately to check if the camera is working
        print("Attempting to get initial frame...")
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            print("Initial frame received successfully")
        except Exception as e:
            print(f"Error getting initial frame: {e}")
            print("Will continue trying in the loop...")
        
        frame_count = 0
        last_print_time = time.time()
        
        while not stop_event.is_set():
            try:
                # Wait for frames with a timeout
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print("No frames received in this iteration")
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Print frame information periodically
                frame_count += 1
                current_time = time.time()
                if current_time - last_print_time >= 5:
                    print(f"Received {frame_count} frames in the last 5 seconds")
                    print(f"Color image shape: {color_image.shape}, Depth image shape: {depth_image.shape}")
                    last_print_time = current_time
                    frame_count = 0
                
                # Store the color and depth images in the detection_results dictionary
                detection_results['color_image'] = color_image
                detection_results['depth_image'] = depth_image
                detection_results['timestamp'] = time.time()
                
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
                time.sleep(0.01)  # Reduced sleep time to 10ms
                
            except Exception as e:
                print(f"Error in realtime_detect loop: {e}")
                time.sleep(1)  # Sleep longer on error
            
    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()

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
    
    # Draw objects inside the board with simple dots
    colors = {
        'circle': (0, 255, 0),  # Green
        'small': (0, 255, 255),  # Yellow
        'big': (255, 255, 0)    # Cyan
    }
    
    for class_name, detection in best_inside_board.items():
        if detection:
            # Get the center point
            center_x, center_y = detection['position_2d']
            
            # Draw a dot at the center point
            color = colors[class_name]
            cv2.circle(visualization_image, (int(center_x), int(center_y)), 5, color, -1)
            
            # Add a label
            cv2.putText(visualization_image, class_name.capitalize(), 
                        (int(center_x) + 10, int(center_y)), 
                        font, font_scale, color, thickness)
    
    # Draw boxes
    for i, box in enumerate(best_boxes):
        points = box['box_points'].astype(np.int32)
        cv2.polylines(visualization_image, [points], True, (255, 0, 0), thickness)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        cv2.putText(visualization_image, f"Box {i+1}", (center_x, center_y), font, font_scale, (255, 0, 0), thickness)
    
    return best_board, best_inside_board, best_boxes, coords_3d_dict, visualization_image 