# Object Detection Library for RealSense Camera

This library provides functions for object detection using a YOLO model with a RealSense camera. It can detect four classes of objects: big, box, circle, and small, but focuses on retrieving the highest confidence detections for big, circle, and small objects.

## Requirements

- Python 3.6+
- Intel RealSense SDK 2.0
- Dependencies listed in `requirements.txt`

## Installation

1. Install Intel RealSense SDK 2.0 from [https://github.com/IntelRealSense/librealsense/releases](https://github.com/IntelRealSense/librealsense/releases)

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install pyrealsense2:
   ```
   pip install pyrealsense2
   ```

## Usage

The library provides the following functions:

### Initialize Camera and Model

```python
from object_detection_lib import initialize_camera, initialize_model

# Initialize camera
if initialize_camera():
    print("Camera initialized successfully")

# Initialize model
if initialize_model("path/to/mennys_model.pt"):
    print("Model initialized successfully")
```

### Capture and Detect Objects

```python
from object_detection_lib import capture_frame, detect_objects

# Capture a frame
color_image, depth_image = capture_frame()

# Detect objects
detections = detect_objects(color_image, depth_image)
```

### Get Best Detection for a Class

```python
from object_detection_lib import get_best_detection

# Get best detection for a specific class (only supports 'big', 'circle', 'small')
best_detection = get_best_detection("circle")
if best_detection:
    print(f"Position: {best_detection['position_2d']}")
    print(f"Confidence: {best_detection['confidence']}")
    print(f"Angle: {best_detection['angle']}")
    print(f"Depth: {best_detection['depth']}")
```

### Convert 2D Position to 3D Coordinates

```python
from object_detection_lib import get_3d_coordinates

# Convert 2D position and angle to 3D coordinates
position_2d_angle = (x, y, angle)  # x, y in pixels, angle in degrees
# Optionally provide depth value from detection
coords_3d = get_3d_coordinates(position_2d_angle, depth=depth_value)
if coords_3d:
    print(f"3D coordinates: {coords_3d}")  # (x, y, z, angle)
```

### Close Camera

```python
from object_detection_lib import close_camera

# Always close the camera when done
close_camera()
```

## Example

See `example_usage.py` for a complete example of how to use the library.

## Notes

- The library stores detection results in four global lists: `big_objects`, `box_objects`, `circle_objects`, and `small_objects`.
- Each detection includes position (2D), depth, confidence, angle, and bounding box coordinates.
- The `get_best_detection()` function only returns the highest confidence detection for 'big', 'circle', and 'small' classes.
- The 3D coordinates are in the camera's coordinate system (meters).
- The depth value in detections corresponds to the depth at the center point of the detected object. 