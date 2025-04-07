## Calibration

To begin calibration, run:
```bash
python base_camera_pos.py
```
During calibration:

* The robot arm will automatically move through 8 distinct positions

* For I don't know what reason, shaking your hand behind AprilTag will make the detection of it more smooth

After successful calibration:

* **The base_camera matrix will be automatically saved as base_camera_pos.npz in your working directory**

* You'll see three key outputs in your terminal:

  - gripper_tag: The transformation matrix from tag to gripper

  - base_camera: The camera_to_base transformation

The camera_base matrix follows this standard format:
````
[[R T]
[0 1]]
````
where R is the rotation matrix and T is the translation vector.

You can use T component of `gripper_tag` and `base_camera` and human eye measure for a quick validation. 
