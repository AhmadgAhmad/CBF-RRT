from cbf import QPcontroller
from robot import iRobot

"""
Created on Mon Feb 4 14:15:20 2019

@author: Zachary Serlin
"""

#robot
def steer(current,sampled,obstacles,end_time,delta_t):
    #delta_t = integrate over #s
    #current = initial state [x1,x2,theta]
    #x_sampled = #sampled data
    #control_v = # m/s tranlsational velocity
    #control_w = # rad/s angular velocitiy
    x_obs = obstacles[0:2]
    r = obstacles[2]

    controller_obj = QPcontroller()
    robot_obj = iRobot()

    for i in range(0,end_time):
        control_v, control_w = controller_obj.generate_control(x_current,x_sampled,x_obs,r)
        x_new = robot_obj.state_update(x_current,control_v,control_w,delta_t)
