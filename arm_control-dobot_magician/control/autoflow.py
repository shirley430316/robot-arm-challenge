import sys
import os
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pydobotplus'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "yolo_ws"))

from pydobotplus import dobotplus as dobot
from mydobot import MyDobot
from get_co import get_co, initialize_system, close_camera
import use_transform_matrix

try:
    # Initialize the camera and model system first
    if not initialize_system():
        print("Failed to initialize camera and model system")
        sys.exit(1)

    device = MyDobot()

    #get three destinations and safe point and zlimit
    a=input("Move it to the table:")
    zlimit=device.get_pose().position.z
    a=input("Move it to the safe point:")
    SAFE_CURVE=[device.get_pose().position.x,device.get_pose().position.y,device.get_pose().position.z,device.get_pose().position.r]
    a=input("Move it to the small destination:")
    des1=[device.get_pose().position.x,device.get_pose().position.y,device.get_pose().position.z,device.get_pose().position.r]
    a=input("Move it to the big destination:")
    des2=[device.get_pose().position.x,device.get_pose().position.y,device.get_pose().position.z,device.get_pose().position.r]
    a=input("Move it to the circle destination:")
    des3=[device.get_pose().position.x,device.get_pose().position.y,device.get_pose().position.z,device.get_pose().position.r]


    #get coordinates from the camera
    result=get_co()
    print(result)

    trans=use_transform_matrix.load_transform_matrix("base_camera_result.npy")

    while(result is not None):
        # define the nail destination
        if result[0]=="circle":
            nail=np.dot(trans, (np.array(np.array(result[1])*1000)*1000))
        else:
            nail=np.dot(trans, (np.array(np.array(result[1])*1000) + np.array(np.array(result[2])*1000)) /2)
        device.safe_jump_to(safe_curve=[SAFE_CURVE], 
                           destination=[nail[0],nail[1],zlimit,angle] ,
                           wait=True)
        device.grip(True, wait_time=2)
        #read angle
        angle=1
        
        if result[0] == "small":
            device.safe_jump_to(safe_curve=[SAFE_CURVE],
                           destination=[des1[0], des1[1], des1[2], des1[3]], 
                           wait=True)

        elif result[0] == "big":
            device.safe_jump_to(safe_curve=[SAFE_CURVE],
                           destination=[des2[0], des2[1], des2[2], des2[3]], 
                           wait=True)

        elif result[0] == "circle":
            device.safe_jump_to(safe_curve=[SAFE_CURVE],
                           destination=[des3[0], des3[1], des3[2], des3[3]], 
                           wait=True)
        
        result=get_co()
        print(result)
        print(np.dot(trans, (np.array(result[1]) + np.array(result[2])) *500))
        time.sleep(5)
        device.grip(False)
        print(device.get_alarms())

    device.close()
finally:
    # Always close the camera when done
    close_camera()

