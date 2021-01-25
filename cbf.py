#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gurobipy import *
import math
import numpy as np

"""
Created on Thu Jan 24 21:28:39 2019

@author: Guang Yang
"""

class QPcontroller:
    def __init__(self):
        self.k_cbf = 2.0 #CBF coefficient
        self.epsilon = 0.1#Finite time CLF coefficient
        self.m = Model("CBF_CLF_QP")
        self.num_of_states = 3
        self.num_of_control_inputs = 2
        self.v_upper_lim = 0.45 # From Create Autonomy
        self.v_lower_lim = -0.45
        self.w_upper_lim = 4.25
        self.w_lower_lim = -4.25

        self.v_obs_1 = 0 #obstacle dynamics in x1 direction
        self.v_obs_2 = 0 #obstacle dynamics in x2 direction
        self.v_r = 0  #obstacle dynamics in change of radius



    def generate_control(self,x_current,x_sampled,x_obs,r):
        '''
        Input:
            x_current   Robot State [x1,x2,theta]
            x_obs       Center of Circular Obstacle [x1,x2]
            r           Radius of Obstacle
            x_sampled   Sampled Desired Location [x1,x2]
        Output:
            v Tranlational Velocity Control Input
            w Angular Velocity Control Input
        '''
        x1 = x_current[0]
        x2 = x_current[1]
        x3 = x_current[2]
        x1s = x_sampled[0]
        x2s = x_sampled[1]
        x3s = x_sampled[2]

        #x_norm = np.linalg.norm([x1-x1s,x2-x2s])

        #Lyapunov Function
        V = (x1-x1s)**2+(x2-x2s)**2 + ((x1-x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5 + math.cos(x3-x3s))**2 + ((x2-x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5 + math.sin(x3-x3s))**2

        # Partial derivatives for calculating time derivative
        partial_V_x1 = 2*(x1-x1s) + 2*(-(x1-x1s)**2/((x1-x1s)**2+(x2-x2s)**2)**1.5+1/((x1-x1s)**2+(x2-x2s)**2)**0.5)*((x1-x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5-math.cos(x3-x3s))-(2*(x1-x1s)*(x2-x2s)*((x2-x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5)-math.sin(x3-x3s))/((x1-x1s)**2+(x2-x2s)**2)**1.5

        partial_V_x2 = 2*(x2-x2s) + 2*(-(x2-x2s)**2/((x1-x1s)**2+(x2-x2s)**2)**1.5+1/((x1-x1s)**2+(x2-x2s)**2)**0.5)*((x2-x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5-math.sin(x3-x3s))-(2*(x1-x1s)*(x2-x2s)*((x1-x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5)-math.cos(x3-x3s))/((x1-x1s)**2+(x2-x2s)**2)**1.5

        partial_V_x3 = -2*math.cos(x3-x3s)*((x2-x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5-math.sin(x3-x3s))+2*((x1-x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5-math.cos(x3-x3s))*math.sin(x3-x3s)



        self.v = self.m.addVar(lb=self.v_lower_lim, ub=self.v_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Translation_Velocity")
        self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")
        #self.v = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Control_Translation_Velocity")
        #self.w = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")
        # Initialize Cost Function
        self.cost_func = self.v*self.v+self.w*self.w
        self.m.setObjective(self.cost_func,GRB.MINIMIZE)

        # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0
        self.m.addConstr((2*(x_current[0]-x_obs[0])*math.cos(x_current[2])*self.v
                            +2*(x_current[1]-x_obs[1])*math.sin(x_current[2])*self.v)
                            +2*(x_current[0]-x_obs[0])*self.v_obs_1
                            +2*(x_current[1]-x_obs[1])*self.v_obs_2
                            -2*r*self.v_r>= -self.k_cbf*((x_current[0]-x_obs[0])**2+(x_current[1]-x_obs[1])**2-r**2),"CBF_constraint")


        self.m.addConstr(partial_V_x1*math.cos(x3)*self.v + partial_V_x2*math.sin(x3)*self.v + partial_V_x3*self.w<=-self.epsilon*V,"CLF_constraint")
        #self.m.addConstr(partial_V_x1*math.cos(x3)*self.v + partial_V_x2*math.sin(x3)*self.v + partial_V_x3*self.w <= -0.2,"CLF_constraint")

        #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        self.solution = self.m.getVars()

        # get final decision variables
        self.control_v = self.solution[0].x
        self.control_w = self.solution[1].x


        # For debuging only, save model to view constraints etc.
        self.m.write("qp_model.lp")

        return self.control_v, self.control_w, V
