from cbf import QPcontroller
from robot import iRobot
from debug_plot import Debug_plot
import matplotlib.pyplot as plt
import math

"""
Created on Mon Feb 4 14:15:20 2019

@author: Zachary Serlin
"""

#robot
if __name__ == "__main__":
    x1_traj = []
    x2_traj = []
    x3_traj = []

    total_steps = 100
    delta_t = 0.2
    x_current = [0,1,0] #initial state [x1,x2,theta]
    x_sampled = [2,2,0] #sampled data
    #control_v = 1 # 2m/s tranlsational velocity
    #control_w = 0 # 0rad/s angular velocitiy
    x_obs = [1,0]
    obs_r = 1.0

    robot_obj = iRobot()

    for step in range(total_steps):
        x1_traj.append([x_current[0]])
        x2_traj.append([x_current[1]])
        x3_traj.append([x_current[2]])
        controller_obj = QPcontroller()

        control_v, control_w, V= controller_obj.generate_control(x_current,x_sampled,x_obs,obs_r)
        x_current = robot_obj.state_update(x_current,control_v,control_w,delta_t)

    # Plot simulated Robot Trajectory
    fig, ax = plt.subplots()
    ax.plot(x1_traj,x2_traj,'b-',x_sampled[0],x_sampled[1],'gx')
    Dynamics_obstacle = plt.Circle((x_obs[0], x_obs[1]), obs_r, color='r')
    ax.set_xlim((-1, 3))
    ax.set_ylim((-1, 3))
    ax.add_artist(Dynamics_obstacle)
    plt.show()


    debug_obj = Debug_plot()
    debug_obj.plot_CLF(x1_traj,x2_traj,x3_traj,delta_t,x_sampled[0],x_sampled[1])
