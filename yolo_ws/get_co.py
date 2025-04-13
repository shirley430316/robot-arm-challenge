import sys
import os
import time
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# Add the path to the yolo_ws directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the object detection functions
from object_detection_lib import (
    initialize_camera,
    initialize_model,
    detect_objects,
    close_camera,
    pipeline,
    get_3d_coordinates
)

# Global variables
camera = None
model = None
pipeline = None
align = None

def is_point_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    Args:
        point: tuple of (x, y) coordinates
        polygon: numpy array of shape (N, 2) containing polygon vertices
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i
    
    return inside

def save_camera_frame():
    """
    Capture a frame from the camera and save it to the input directory.
    Returns the path to the saved image and the depth frame.
    """
    global pipeline, align
    
    if pipeline is None or align is None:
        print("Camera not initialized")
        return None, None
        
    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Failed to get frames")
            return None, None
            
        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create input directory if it doesn't exist
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Save color image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(input_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(image_path, color_image)
        
        return image_path, depth_frame
        
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None, None

def get_co():
    """
    Get coordinates of detected objects from camera frame.
    Returns a list in format: ["class", [x1,y1,z1,1], [x2,y2,z2,1]] for rectangles
    or ["circle", [x,y,z,1]] for circles.
    Only returns objects detected inside the board.
    Returns None if no board is detected or no objects are found inside the board.
    """
    global pipeline, align, model
    
    # Initialize camera if not already initialized
    if pipeline is None:
        success, pipe = initialize_camera()
        if not success or pipe is None:
            print("Failed to initialize camera")
            return None
        pipeline = pipe
        align = rs.align(rs.stream.color)
    
    # Initialize model if not already initialized
    if not initialize_model():
        print("Failed to initialize model")
        return None
    
    # First save a camera frame
    image_path, depth_frame = save_camera_frame()
    if image_path is None or depth_frame is None:
        print("Failed to capture frame")
        return None
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from {image_path}")
            return None
            
        # Run detection with parameters from test_model.py
        results = model.predict(
            source=image,
            conf=0.35,    # Confidence threshold
            iou=0.2,      # NMS IOU threshold
            max_det=30,   # Maximum detections per image
            verbose=False # Reduce output information
        )
        
        if len(results) == 0:
            print("No detections found")
            return None
            
        result = results[0]
        if not hasattr(result, 'obb') or result.obb is None:
            print("No objects detected")
            return None
            
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # First find the board
        board = None
        board_conf = 0
        
        # Process all detections to find board
        for i in range(len(result.obb.cls)):
            cls = int(result.obb.cls[i])
            conf = float(result.obb.conf[i])
            class_name = result.names[cls]
            
            if class_name == 'board':
                if conf > board_conf:
                    board_conf = conf
                    points = result.obb.xyxyxyxy[i].cpu().numpy()  # Convert to numpy array
                    board = points.reshape((-1, 2))
        
        if board is None:
            print("No board detected")
            return None
            
        # Draw board on visualization image
        cv2.polylines(vis_image, [board.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Now find objects inside the board
        objects = []  # List of (class_name, confidence, points)
        
        for i in range(len(result.obb.cls)):
            cls = int(result.obb.cls[i])
            conf = float(result.obb.conf[i])
            class_name = result.names[cls]
            
            if class_name in ['big', 'small', 'circle']:
                points = result.obb.xyxyxyxy[i].cpu().numpy()  # Convert to numpy array
                points = points.reshape((-1, 2))
                
                # Check if object center is inside board
                center = np.mean(points, axis=0)
                if is_point_inside_polygon((center[0], center[1]), board):
                    objects.append((class_name, conf, points))
        
        if not objects:
            print("No objects found inside board")
            # Save visualization image with only board
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(output_path, vis_image)
            print(f"Detection visualization saved to: {output_path}")
            return None
            
        # Sort objects by priority (big > small > circle) and confidence
        priority = {'big': 3, 'small': 2, 'circle': 1}
        objects.sort(key=lambda x: (priority[x[0]], x[1]), reverse=True)
        
        # Get the best object
        best_class, _, points = objects[0]
        
        # Draw the best object on visualization image
        cv2.polylines(vis_image, [points.astype(np.int32)], True, (255, 0, 0), 2)
        cv2.putText(vis_image, best_class, (int(points[0][0]), int(points[0][1])-10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save visualization image
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"Detection visualization saved to: {output_path}")
        
        if best_class in ['big', 'small']:
            # For rectangles, get midpoints of short edges
            edges = []
            for i in range(4):
                next_i = (i + 1) % 4
                edge = points[next_i] - points[i]
                length = np.linalg.norm(edge)
                edges.append((i, next_i, length))
            
            # Sort edges by length
            edges.sort(key=lambda x: x[2])
            
            # Get midpoints of short edges (first two edges are shortest)
            midpoints_2d = []
            for i, next_i, _ in edges[:2]:
                midpoint = (points[i] + points[next_i]) / 2
                midpoint = (int(midpoint[0]), int(midpoint[1]))  # Convert to integer tuple
                midpoints_2d.append(midpoint)
                # Draw midpoints on visualization image
                cv2.circle(vis_image, midpoint, 3, (0, 0, 255), -1)
            
            # Get 3D coordinates for both midpoints
            coords_3d = []
            valid_depth = None
            
            # First pass: try to get valid depth for both midpoints
            for midpoint in midpoints_2d:
                x, y = midpoint
                try:
                    # Get depth value with error handling
                    depth = depth_frame.get_distance(int(x), int(y))
                    print(f"Depth at ({x}, {y}): {depth}")
                    
                    # Check if depth is valid
                    if depth <= 0:
                        print(f"Invalid depth value at ({x}, {y}): {depth}")
                        # Try to get depth from surrounding points
                        depth_values = []
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                try:
                                    d = depth_frame.get_distance(x + dx, y + dy)
                                    if d > 0:
                                        depth_values.append(d)
                                except:
                                    continue
                        
                        if depth_values:
                            depth = np.median(depth_values)
                            print(f"Using median depth from surrounding points: {depth}")
                        else:
                            print("No valid depth values found in surrounding points")
                            depth=0.5
                            continue
                    
                    # Store the first valid depth we find
                    if valid_depth is None and depth > 0:
                        valid_depth = depth
                    
                    point_3d = get_3d_coordinates((x, y, 0), depth, pipeline)
                    if point_3d is not None:
                        # Keep original millimeter units from RealSense
                        coords_list = [point_3d[0], point_3d[1], point_3d[2], 0.001]
                        coords_3d.append(coords_list)
                except Exception as e:
                    print(f"Error getting 3D coordinates for midpoint ({x}, {y}): {e}")
                    continue
            
            # Second pass: if we have a valid depth but not enough 3D coordinates, use the valid depth for all midpoints
            if len(coords_3d) < 2 and valid_depth is not None:
                print(f"Using valid depth {valid_depth} for all midpoints")
                for midpoint in midpoints_2d:
                    x, y = midpoint
                    point_3d = get_3d_coordinates((x, y, 0), valid_depth, pipeline)
                    if point_3d is not None:
                        # Keep original millimeter units from RealSense
                        coords_list = [point_3d[0], point_3d[1], point_3d[2], 0.001]
                        coords_3d.append(coords_list)
            
            # If we have at least one valid coordinate, use it
            if len(coords_3d) >= 1:
                # If we only have one coordinate, duplicate it
                if len(coords_3d) == 1:
                    print("Only one valid 3D coordinate found, duplicating it")
                    coords_3d.append(coords_3d[0])
                
                # Save final visualization image with midpoints
                cv2.imwrite(output_path, vis_image)
                
                # Create a list to store the result
                result = [best_class]
                
                # Add coordinates to the result
                for coord in coords_3d:
                    # Convert each coordinate to a NumPy array
                    coord_array = np.array(coord)
                    result.append(coord_array)
                
                # Print the result for debugging
                print("Result from get_co:", result)
                
                return result
            else:
                print("No valid 3D coordinates found")
                return None
                
        else:  # circle
            # Get center point
            center = np.mean(points, axis=0)
            center = (int(center[0]), int(center[1]))  # Convert to integer tuple
            x, y = center
            
            # Draw center point on visualization image
            cv2.circle(vis_image, center, 3, (0, 0, 255), -1)
            
            # Get 3D coordinates for circle
            try:
                # Get depth value with error handling
                depth = depth_frame.get_distance(int(x), int(y))
                print(f"Depth at circle center ({x}, {y}): {depth}")
                
                # Check if depth is valid
                if depth <= 0:
                    print(f"Invalid depth value at circle center ({x}, {y}): {depth}")
                    # Try to get depth from surrounding points
                    depth_values = []
                    for dx in np.arange(-0.03, 0.04,0.001):
                        for dy in np.arange(-0.03, 0.04,0.001):
                            try:
                                d = depth_frame.get_distance(int(x) + dx, int(y) + dy)
                                if d > 0:
                                    depth_values.append(d)
                            except:
                                continue
                    
                    if depth_values:
                        depth = np.median(depth_values)
                        print(f"Using median depth from surrounding points: {depth}")
                    else:
                        print("No valid depth values found in surrounding points")
                        return None
                
                point_3d = get_3d_coordinates((x, y, 0), depth, pipeline)
                
                if point_3d is not None:
                    # Keep original millimeter units from RealSense
                    coords_list = [point_3d[0], point_3d[1], point_3d[2], 0.001]
                    # Save final visualization image with center point
                    cv2.imwrite(output_path, vis_image)
                    
                    # Create a list to store the result
                    result = [best_class]
                    
                    # Convert coordinates to a NumPy array
                    coord_array = np.array(coords_list)
                    result.append(coord_array)
                    
                    # Print the result for debugging
                    print("Result from get_co:", result)
                    
                    return result
                else:
                    print("Failed to get 3D coordinates for circle center")
                    return None
            except Exception as e:
                print(f"Error getting 3D coordinates for circle center ({x}, {y}): {e}")
                return None
        
        return None
        
    except Exception as e:
        print(f"Error in get_co: {e}")
        import traceback
        traceback.print_exc()
        return None

def initialize_model():
    """Initialize the YOLO model."""
    global model
    try:
        if model is None:
            model_path = os.path.join(os.path.dirname(__file__), "model", "mennys_model3.pt")
            print(f"Loading model from: {model_path}")
            model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def initialize_system():
    """
    Initialize camera and model system.
    Returns True if successful, False otherwise.
    """
    global pipeline, align
    
    try:
        # Initialize camera
        success, pipe = initialize_camera()
        if not success or pipe is None:
            print("Failed to initialize camera")
            return False
        pipeline = pipe
        align = rs.align(rs.stream.color)
        
        # Initialize model
        if not initialize_model():
            print("Failed to initialize model")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False

def close_camera():
    """Close the camera pipeline."""
    global pipeline
    if pipeline:
        pipeline.stop()
        pipeline = None

if __name__ == "__main__":
    # Initialize system
    if not initialize_system():
        print("Failed to initialize system")
        exit(1)
    
    try:
        # Test detection
        result = get_co()
        print("Detection result:", result)
        
    finally:
        # Clean up
        close_camera() 