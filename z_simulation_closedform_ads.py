import matplotlib.pyplot as plt
import os
import math
import numpy as np
import scipy
from numpy import linalg as LA
from gurobipy import *

from cbf.obstacle import Sphere, Ellipsoid
from cbf.agent import Agent
from cbf.goal import Goal
from cbf.simulation import Simulation
from cbf.params import Params
# from scipy.integrate import solve_ivp



"""
Created on Mon Oct 24 9:18 PM

@author: Guang Yang

This code is a rewritten version of the original CBF-RRT paper with
linear dynamics
"""
class Obstacle_Sphere(object):
    def __init__(self, center, radius):
        self.T = 1.0  #Integration Length
        self.N = 20 # Number of Control Updates
        self.center=center
        self.radius=radius

    def h(self,x):
        return LA.norm(x-self.center,2)**2-self.radius**2

    def gradh(self,x):
        return 2*(x-self.center)

    def hdot(self,x,xdot):
        return self.gradh(x).transpose().dot(xdot)

def fun_derivative_trajectory(x,dx,f,gradf):
    fx=np.apply_along_axis(f,0,x)
    gradfx=np.apply_along_axis(gradf,0,x)
    dfx=np.sum(gradfx*dx,0)

    #plot fx and dfx
    t_span = np.linspace(0,7.0,100)

    fig, ax = plt.subplots()
    ax.plot(t_span, -fx,'r',label="-h")
    #ax.plot(t_span, gradfx[0,:],'g')
    ax.plot(t_span, dfx,'b',label="h_dot")
    ax.set_xlabel("Time")
    ax.legend()

    plt.show()

class MA_CBF_RRT:
    def __init__(self, initial_state, obstacle_list):
        self.t0 = 0 # Starting time
        self.T = 1.0  #Integration Length
        self.N = 10 # Number of Control Updates
        self.y0 = initial_state.flatten()
        self.num_agents = len(initial_state[0])
        self.k_cbf = 1.0 #CBF coefficient
        self.p_cbf = 1 #CBF constraint power
        self.x_obstacle = obstacle_list
        self.u1_lower_lim = np.repeat(-5,self.num_agents)
        self.u1_upper_lim = np.repeat( 5,self.num_agents)
        self.u2_lower_lim = np.repeat(-5,self.num_agents)
        self.u2_upper_lim = np.repeat( 5,self.num_agents)

        self.u1_traj = np.zeros(shape=(0,0))
        self.u2_traj = np.zeros(shape=(0,0))
        self.x1_traj = np.zeros(shape=(0,0))
        self.x2_traj = np.zeros(shape=(0,0))

        self.cbf_traj = np.zeros(shape=(0,0))
        self.hdot_traj = np.zeros(shape=(0,0))
        self.h_traj = np.zeros(shape=(0,0))

        self.params = Params()
        self.params.step_size = self.T/self.N


    def controller(self,x_current,u_ref):
        sim = Simulation()
        for i in range(self.num_agents):
            x = x_current[i]
            y = x_current[self.num_agents+i]
            ur = u_ref[i,0:2]
            ur = np.tile(ur, (self.N,1))
            sim.add_agent(Agent((x,y),ur.T))

        for obst in self.x_obstacle:
            sim.add_obstacle(obst)
        
        x,u = sim.initiate(self.N)

        # Reshape
        x = np.reshape(x, (2*self.num_agents,self.N), order='F')
        u = np.reshape(u, (2*self.num_agents,self.N), order='F')

        return x,u

    def get_solArr_val(self, indices):
        return [self.solution[i].x for i in indices]


    def motion_planning(self,u_ref):
        x_current = np.expand_dims(self.y0, axis=1)
        x = np.zeros((2*self.num_agents,0))
        u = np.zeros((2*self.num_agents,0))
        delta_t = self.T/self.N

        x,u = self.controller(x_current[:,0],u_ref)
        
        return (x,u)

    def plot_traj(self,x,u):
        t_span = np.linspace(0,self.T,self.N)

        fig, ax = plt.subplots()

        circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]),
        obstacle_list[0][2], color='r',alpha=0.2)
        ax.add_artist(circle)
        ax.plot(x[0,:], x[1,:])
        ax.set_xlim(-1,5)
        ax.set_ylim(-1,5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

if __name__ == "__main__":
    initial_state = np.array([[1.0],[1.0]])
    obstacle_list = [[2.9,2.6,0.5]]
    u_ref = [0.5,0.5]

    CBFRRT_Planning = CBF_RRT(initial_state, obstacle_list)
    x, u= CBFRRT_Planning.motion_planning(u_ref)
    CBFRRT_Planning.plot_traj(x,u)

    #Sphere = Obstacle_Sphere([obstacle_list[0][0],obstacle_list[0][1]],obstacle_list[0][2])
    #fun_derivative_trajectory(x,u,Sphere.h,Sphere.gradh)
