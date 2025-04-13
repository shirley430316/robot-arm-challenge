import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pydobotplus'))

from pydobotplus import dobotplus as dobot
from mydobot import MyDobot

device = MyDobot()

current = device.get_pose().position
l=[current.x,current.y,current.z,current.r,]

print(l)

device.close()