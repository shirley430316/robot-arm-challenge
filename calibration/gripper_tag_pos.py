import numpy as np

def get_gripper_tag_transform_matrix():
    return np.array(([-1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 150],
                         [0, 0, 0, 1]))