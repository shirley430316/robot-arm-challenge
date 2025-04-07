import sys
import os
import time
import cv2
import pyrealsense2 as rs
from threading import Thread, Event
import numpy as np

# Add the path to the yolo_ws directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the object detection functions
from object_detection_lib import (
    initialize_camera,
    initialize_model,
    find_best_objects,
    realtime_detect,
    close_camera
)

def get_co(model_path="yolo_ws/model/model.pt", output_dir="yolo_ws/output"):
    """
    Initialize camera and model, detect objects, and return 3D coordinates with rotation.
    
    Args:
        model_path: Path to the YOLO model weights file
        output_dir: Directory to save output images
        
    Returns:
        list: A list containing sublists for each detected object:
              [[[x, y, z, 1], [rotation]], ...]
              The order is: big, circle, small, box1, box2, box3
              If an object is not detected, its entry will be None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model
    if not initialize_model(model_path):
        print("Model initialization failed")
        return [None] * 6
    
    # Initialize the camera
    if not initialize_camera():
        print("Camera initialization failed")
        return [None] * 6
    
    # Create a stop event for the detection thread
    stop_event = Event()
    
    # Create a dictionary to store detection results
    detection_results = {}
    
    try:
        # Start real-time detection in a separate thread
        detection_thread = Thread(target=realtime_detect, args=(None, stop_event, detection_results))
        detection_thread.daemon = True
        detection_thread.start()
        
        print("\n=== Object Detection ===")
        print("Detecting objects...")
        
        # Wait for detection results
        max_wait_time = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check if we have new detection results
            if 'timestamp' in detection_results and time.time() - detection_results['timestamp'] < 5:
                # Get the latest color and depth images
                color_image = detection_results.get('color_image')
                depth_image = detection_results.get('depth_image')
                
                if color_image is not None and depth_image is not None:
                    # Find the best objects
                    best_board, best_inside_board, best_boxes, coords_3d_dict, visualization_image = find_best_objects(
                        color_image, depth_image
                    )
                    
                    # Save the visualization image
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(output_path, visualization_image)
                    print(f"Detection image saved to {output_path}")
                    
                    # Extract coordinates in the requested format
                    result = [None] * 6
                    
                    # Get coordinates for objects inside the board
                    if 'big' in coords_3d_dict:
                        coords = coords_3d_dict['big']
                        result[0] = [[coords[0], coords[1], coords[2], 1], [coords[3]]]
                    
                    if 'circle' in coords_3d_dict:
                        coords = coords_3d_dict['circle']
                        result[1] = [[coords[0], coords[1], coords[2], 1], [coords[3]]]
                    
                    if 'small' in coords_3d_dict:
                        coords = coords_3d_dict['small']
                        result[2] = [[coords[0], coords[1], coords[2], 1], [coords[3]]]
                    
                    # Get coordinates for boxes
                    for i in range(3):
                        box_key = f'box_{i+1}'
                        if box_key in coords_3d_dict:
                            coords = coords_3d_dict[box_key]
                            result[i+3] = [[coords[0], coords[1], coords[2], 1], [coords[3]]]
                    
                    # Print 3D coordinates
                    print("\n=== 3D Coordinates ===")
                    for obj_name, coords in coords_3d_dict.items():
                        print(f"{obj_name}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}, rotation: {coords[3]:.3f})")
                    
                    return result
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
        
        print("Timeout waiting for detection results")
        return [None] * 6
            
    except Exception as e:
        print(f"Error during detection: {e}")
        return [None] * 6
    finally:
        # Stop the detection thread
        stop_event.set()
        detection_thread.join(timeout=1)
        
        # Close the camera
        close_camera()
        print("Cleanup completed")

def show_detection_image(image_path=None):
    """
    Display a detection image from the output directory.
    
    Args:
        image_path: Path to the image to display. If None, displays the most recent image.
    """
    output_dir = "yolo_ws/output"
    
    if image_path is None:
        # Find the most recent image in the output directory
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist")
            return
        
        image_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        if not image_files:
            print(f"No images found in {output_dir}")
            return
        
        # Sort by modification time (most recent first)
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        image_path = os.path.join(output_dir, image_files[0])
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist")
        return
    
    # Read and display the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}")
        return
    
    cv2.imshow("Detection Image", image)
    print(f"Displaying image: {image_path}")
    print("Press any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Get coordinates
    coords = get_co()
    
    # Print the results
    print("\n=== Results ===")
    print(f"Big coordinates: {coords[0]}")
    print(f"Circle coordinates: {coords[1]}")
    print(f"Small coordinates: {coords[2]}")
    print(f"Box 1 coordinates: {coords[3]}")
    print(f"Box 2 coordinates: {coords[4]}")
    print(f"Box 3 coordinates: {coords[5]}")
    
    # Show the detection image
    show_detection_image() 