from inverse_robot import iRobot_inv
from robot import iRobot

import numpy as np
import matplotlib.pyplot as plt

"""
Created on Mon Feb 4 14:15:20 2019

@author: Zachary Serlin
"""

#robot

robot_obj_inv = iRobot_inv()
end_time = 10
delta_t = .1
x_new=[0.00,0.00,0.00]
x_save = []
y_save = []
controlsv = np.multiply(0.2,range(end_time))
controlsw = np.multiply(0.1,range(end_time))
print(x_new)
for i in range(0,end_time):
    x_save.append(x_new[0])
    y_save.append(x_new[1])
    control_v = controlsv[i]
    control_w = controlsw[i]
    #control_v = 0.2*i
    #control_w = 0.01*i
    x_new = robot_obj_inv.state_update(x_new,control_v,control_w,delta_t)
    print(x_new)
x_save.append(x_new[0])
y_save.append(x_new[1])

robot_obj = iRobot()
x2_new=x_new
x2_save = []
y2_save = []
print(x_new)
for i in range(0,end_time):
    j = (end_time-1)-i
    x2_save.append(x2_new[0])
    y2_save.append(x2_new[1])
    #control_v = 0.2*j
    #control_w = 0.01*j
    control_v = controlsv[j]
    control_w = controlsw[j]
    x2_new = robot_obj.state_update(x2_new,control_v,control_w,delta_t)
    print(x2_new)
#x2_save.append(x2_new[0])
#y2_save.append(x2_new[1])

plt.cla()
plt.scatter(x_save,y_save,c='r')
plt.scatter(x2_save,y2_save,c='b')
plt.grid(True)
plt.show()
