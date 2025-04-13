import sys
import os
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pydobotplus'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "yolo_ws"))

from pydobotplus import dobotplus as dobot
from mydobot import MyDobot
# from get_co import get_co, initialize_system, close_camera
import use_transform_matrix

device = MyDobot()
podition=[]
for i in range(20):
    input("Press Enter to get position")
    p=device.get_pose().position
    podition.append([p.x,p.y,p.z,p.r])

print(podition)
device.close()
