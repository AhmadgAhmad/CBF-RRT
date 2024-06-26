import numpy as np
import math
import matplotlib.pyplot as plt

from gurobipy import *
from function import Function
from dynamics import Dyn, SingleIntegrator, Unicycle, DoubleIntegrator, UnicycleExtended
from goal import Goal
from params import *

class Agent(object):

    def __init__(self, init_pos, instructs, radius=0.5, init_vel=(0,0), dynamics=Dyn.SINGLE_INT, theta=.1, eps=None, k_cbf=1.0):
        """
        init_pos: (Tuple) (x,y) Start location
        radius: (float) Radius of agent
        init_vel: (Tuple) (dx, dy) Initial velocity
        """
        self.params = Params()

        if dynamics is Dyn.SINGLE_INT:
            self.dyn = SingleIntegrator(init_pos)
            self.dyn_enum = Dyn.SINGLE_INT

        elif dynamics is Dyn.UNICYCLE:
            self.dyn = Unicycle(init_pos, theta)
            self.dyn_enum = Dyn.UNICYCLE

        elif dynamics is Dyn.DOUBLE_INT:
            self.dyn = DoubleIntegrator(init_pos)
            self.dyn_enum = Dyn.DOUBLE_INT

        elif dynamics is Dyn.UNICYCLE_EXT:
            self.dyn = UnicycleExtended(init_pos)
            self.dyn_enum = Dyn.UNICYCLE_EXT

        self.init_pos = init_pos
        self.state = self.dyn.cur_state
        self.radius = radius
        self.done = False
        self.eps = self.params.epsilon if eps is None else eps
        self.steps = 0
        self.k_cbf = 1.0

        if type(instructs) is Goal:
            self.goal = instructs
            self.u_ref = None
        else:
            self.goal = None
            self.u_ref = instructs

        # For CBF Constraints
        h_func = lambda a1: (a1.state-self.state).T.dot((a1.state-self.state))[0][0] - (a1.radius+self.radius)**2
        del_h = lambda a1: 2*(a1.state - self.state)
        self.h = Function(h_func, del_h)

        # These fields used for plotting and will be set later
        self.axes = None
        self.circle = None
        self.color = 'g'

    def add_clf(self, m):
        if self.goal is not None:
            return self.goal.add_clf(m,self)

    def add_control(self, m, id):
        u = self.dyn.add_control(m, id)
        self.u = u

    def get_state(self, t_idx=-1):
        return self.dyn.get_state(t_idx)

    def get_x_dot(self, u=None):
        if u is None:
            u = self.u
        return self.dyn.get_x_dot(self.state, u)
    
    def step(self, u, plot=True):
        if u == None:
            u = [x[0].x for x in self.u]
            u = np.array(u)
            u.shape = (len(u),1)
        self.state = self.dyn.step(u)
        self.steps += 1

        if self.goal is None:
            self.done = max(self.u_ref.shape) == self.steps
        else:
            self.done = self.dist_to_goal() <= self.eps

        if plot:
            self.plot()
            plt.pause(.01)

    def stepX(self, num_steps, u, plot=True, printout=True):
        for i in range(num_steps):
            self.step(u, plot)

            if printout:
                self.print_state()

    def dist_to_goal(self):
        return np.sqrt((self.state-self.goal.goal).transpose().dot(self.state-self.goal.goal)[0][0])

    def print_state(self):
        self.dyn.print_state()

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plotCircle(self.state[0], self.state[1], self.radius, ax, self.color)
        traj = self.dyn.trajectory
        plt.plot(traj[0,:],traj[1,:],"-",color=self.color)
        
        plt.plot(self.init_pos[0],self.init_pos[1],"o",color=self.color, markersize=10)
        if self.goal is not None:
            self.goal.plot(ax, color=self.color)

        if isinstance(self.dyn, Unicycle):
            # Plot arrow indicating direction
            plotArrow(self.state[0][0], self.state[1][0], self.dyn.cur_theta, self.radius,ax, color=self.color)            
        elif isinstance(self.dyn, DoubleIntegrator):
            # Plot arrow indicating velocity magnitude
            mag = np.linalg.norm(self.state[2:4,:])
            if mag != 0:
                theta = np.arctan2(self.state[3,0],self.state[2,0])
                basex = self.state[0][0] + np.cos(theta)*self.radius
                basey = self.state[1][0] + np.sin(theta)*self.radius
                plotArrow(basex, basey, theta, mag,ax, color=self.color)

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec

def plotCircle(x, y, size, ax, color='g'):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(math.radians(d)) for d in deg]
    yl = [y + size * math.sin(math.radians(d)) for d in deg]
    
    if ax is None:
        _, ax = plt.subplots()
    
    h_line = ax.plot(xl, yl, "-", color=color, markersize=100)
    ax.axis('equal')

def plotArrow(x, y, theta, radius, ax, color='g'):
    dx = math.cos(theta) * radius
    dy = math.sin(theta) * radius
    ax.arrow(x,y,dx,dy,length_includes_head=True, width=2*radius*.05, head_width=2*radius*.3, fc=color,ec=color)

def main():
    _,ax = plt.subplots()

    pos1 = (0,0)
    a1 = Agent(pos1, dynamics=Dyn.DOUBLE_INT)
    a1.plot()
    plt.pause(1)

    u = (1,1)
    a1.stepX(50,u)

    u = (1,-1)
    a1.stepX(50,u)

    u = (-1,-1)
    a1.stepX(50,u)

    u = (-1, 1)
    a1.stepX(50,u)


if __name__ == '__main__':
    main()
    