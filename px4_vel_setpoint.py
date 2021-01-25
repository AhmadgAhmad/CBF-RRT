"""
PX4 Velcotiy Controls:
Guang Yang

"""

import rospy
import numpy as np
import math
import mavros_msgs
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs import srv
from mavros_msgs.msg import State
import pandas as pd


goal_pose = PoseStamped()
current_pose = PoseStamped()
set_velocity = TwistStamped()
current_state = State()
Kp = 1.0


def pos_msg_to_numpy():
    pose_numpy = np.array([current_pose.pose.position.x,current_pose.pose.position.y]).reshape(2,1)
    return pose_numpy

def numpy_to_vel_msg(cmd_vel):
    set_velocity.twist.linear.x = cmd_vel[0,0]
    set_velocity.twist.linear.y = cmd_vel[1,1]

def pos_sub_callback(pose_sub_data):
    global current_pose
    current_pose = pose_sub_data

def state_callback(state_data):
    global current_state
    current_state = state_data

def hold_z():
    global goal_pose
    goal_pose.pose.position.z = 2.0 # hold at 2m
    error_in_z = goal_pose.pose.position.z - current_pose.pose.position.z
    set_velocity.twist.linear.z = error_in_z * Kp


rospy.init_node('Vel_Control_Node', anonymous = True)
rate = rospy.Rate(5) #publish at 100 Hz
local_position_subscribe = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pos_sub_callback)
setpoint_velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size = 30)
state_status_subscribe = rospy.Subscriber('/mavros/state',State,state_callback)

cmd_vel = np.load('u_traj.npy')
counter = 0



while not rospy.is_shutdown():

    # TO DO: cmd_Vel = get_planning_vel(current_time) Obtain the command velocity from CBF-RRT plans
    #numpy_to_vel_msg(cmd_vel)



    hold_z()
    setpoint_velocity_pub.publish(set_velocity)

    if  current_state.armed:
        set_velocity.twist.linear.x = cmd_vel[0,counter]
        set_velocity.twist.linear.y = cmd_vel[1,counter]
        counter = counter + 1

    # Only for simulation, remove the code below in actual experiment (Use hardware switch for "OFFBOARD")
    if current_state.mode != "OFFBOARD" or not current_state.armed:
        arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        arm(True)
        set_mode = rospy.ServiceProxy('/mavros/set_mode',mavros_msgs.srv.SetMode)
        mode = set_mode(custom_mode = 'OFFBOARD')

        #if mode.success:
        #    rospy.loginfo('Switched to OFFBOARD mode!')
    rate.sleep()
