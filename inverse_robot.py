from matplotlib import pyplot as plt
from math import sin, cos
from casadi import *

"""
Created on Sat Fed 2 14:15:20 2019

@author: Guang Yang

Implemented through CASADI
"""

class iRobot_inv:
    def __init__(self):
        self.x1 = MX.sym('x1')
        self.x2 = MX.sym('x2')
        self.theta = MX.sym('theta')
        self.x = vertcat(self.x1, self.x2, self.theta)
        self.v = MX.sym('v')
        self.w = MX.sym('w')
        self.xdot = vertcat(-1*self.v*cos(self.theta), -1*self.v*sin(self.theta), -1*self.w)

    def state_update(self,x_current,control_v,control_w,delta_t):
        '''
        The state_update function takes in control input u and ZOH time delta_t,
        then it intergrates based on the robot dynamics to get the end
        state
        '''
        # Setup integrator
        self.dae = {'x':self.x, 'p':vertcat(self.v,self.w), 'ode':self.xdot}
        self.opts = {'t0': 0,'tf':delta_t}
        self.F = integrator('F', 'cvodes', self.dae, self.opts)

        # Integrate given dynamics and control
        #self.Fk = self.F(x0=[x_current[0],x_current[1],x_current[2]],p=vertcat(control_v, control_w))
        self.Fk = self.F(x0=x_current,p=vertcat(control_v, control_w))
        x_new = self.Fk['xf']
        return x_new
