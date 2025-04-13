# dobot_magician_control

This is a repository to store the python code for controlling the Dobot [Magician](https://www.dobot-robots.com/products/education/magician.html) robot arm. For ROS1 version, please refer to our another [repo](https://github.com/HKUArmStrong/dobot_magician_ros.git).

## Pre-requisites

- Install the required packages
    ```bash
    conda create -n dobot python=3.10
    conda activate dobot
    pip install -r requirements.txt
    ```

## Available Programs

### Check Port
This program checks the available ports for the Dobot Magician.
 
 
Run the program:
```bash
# tested on ubuntu
python scripts/check_port.py
```

Use the robot arm to achieve nails classification:
need to quote funcitons from yolo_ws and first run 
python base_camera_pos-v2.py
to measure the transform matrix in calibration.
after generated the transform matrix(npy file), can run 
autoflow.py under control directory.

Run the program:
```bash
python scripts/output_coordinates.py
```
