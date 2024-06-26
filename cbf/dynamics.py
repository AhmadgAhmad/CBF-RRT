import sys
import numpy as np
import math
from gurobipy import *
from enum import Enum
import abc
from params import *
import matplotlib.pyplot as plot
import matplotlib

class Dyn(Enum):
    UNICYCLE = 0
    SINGLE_INT = 1
    DOUBLE_INT = 2

class Dynamics(object):

    @abc.abstractmethod
    def __init__(self, init_pos):
        pass

    @abc.abstractmethod
    def get_state(self, t_idx=None):
        pass

    @abc.abstractmethod
    def get_x_dot(self):
        pass

    @abc.abstractmethod
    def add_control(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def print_state(self):
        pass

class UnicycleExtended(Dynamics):
    """
        x_dot(t)     = v.cos(theta)
        y_dot(t)     = v.sin(theta)
        theta_dot(t) = w
        v_dot(t)     = mu
        x_dotdot(t)  = -v.sin(theta).w + mu.cos(theta)
        y_dotdot(t)  = v.cos(theta).w + mu.sin(theta)

        q = [x(t), y(t), theta(t), v(t), x_dot(t), y_dot(t)] The states of the extended dynamics
        """

    def __init__(self, init_pos):
        self.params = Params()
        init_pos = make_column(init_pos)
        self.cur_state = init_pos #Current q(t)

        self.time_step = self.params.step_size
        self.cur_time = 0
        self.time = np.array([0])
        self.trajectory = np.array(self.cur_state)

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:, t_idx]

    def get_x_dot(self,q,u): #still need to define the controls
        w,mu = np.array(u[0]),np.array(u[1])
        q1,q2,q3,q4,q5,q6 = np.array(q[0]),np.array(q[1]),np.array(q[2]),\
                            np.array(q[3]),np.array(q[4]), np.array(q[5])

        q1_dot = np.array(q4*math.cos(q3))
        q2_dot = np.array(q4*math.sin(q3))
        q3_dot = np.array(w)
        q4_dot = np.array(mu)
        q5_dot = np.array((-q4*w*math.sin(q3))+(mu*math.cos(q3)))
        q6_dot = np.array((q4*w*math.cos(q3))+(mu*math.sin(q3)))

        return np.vstack([q1_dot,q2_dot,q3_dot,q4_dot,q5_dot,q6_dot])

    def add_control(self,m,id): #Adds Gurobi optimization variables to solve for controls
        w_ub =  self.params.we_upper_bound
        mu_ub = self.params.mu_upper_bound
        w = m.addVar(lb=-w_ub, ub=w_ub, vtype=GRB.CONTINUOUS, name="omega{}".format(id))
        mu = m.addVar(lb=0, ub=mu_ub, vtype=GRB.CONTINUOUS, name="mu{}".format(id))
        return np.array([[w],[mu]])

    def step(self,u): #To compute the next step of the numerical integration:
        xk = self.cur_state
        x_dot = self.get_x_dot(xk, u)
        dt = self.time_step
        xk_1 = xk + x_dot*dt #The Euler integration

        #The trajectory need to updated here as well -- it won't be returned though.
        self.cur_state = xk_1 #Update the state of the system dyanims
        self.cur_time += self.time_step #Update where we are on the time of the trajectory

        self.time = np.append(self.time,self.cur_time) #accomulate the time vector
        self.trajectory = np.append(self.trajectory,self.cur_state,axis=1)

        return xk_1

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)


class Unicycle(Dynamics):
    """
    Single Integrator dynamics that are transformed to behave like unicycle dynamics
    x(t) = [x, y, theta]'

    .       | cos(theta)  0 |
    x(t) =  | sin(theta)  0 | | v |
            |     0       1 | | w |
    """
    def __init__(self, init_pos, theta=0):
        self.params = Params()
        l = self.params.l 
        
        self.rot_mat = lambda theta: (np.array([[math.cos(theta), -l*math.sin(theta)],[math.sin(theta),l*math.cos(theta)]]))

        init_pos = make_column(init_pos)
        
        self.cur_state = init_pos + l * np.array([[math.cos(theta)],[math.sin(theta)]])
        self.cur_theta = theta

        self.cur_time = 0
        
        self.time = np.array([0])
        self.trajectory = np.array(self.cur_state)

        self.time_step = self.params.step_size

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u):
        return self.rot_mat(self.cur_theta).dot(np.array(u))

    def add_control(self, m, id):
        v_ub = self.params.v_upper_bound
        w_ub = self.params.w_upper_bound
        v = m.addVar(lb=0, ub=v_ub, vtype=GRB.CONTINUOUS, name="vel{}".format(id))
        w = m.addVar(lb=-w_ub, ub=w_ub, vtype=GRB.CONTINUOUS, name="omega{}".format(id))
        return np.array([[v],[w]])

    def step(self, u):
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step

        u = make_column(u)

        self.cur_state = x1
        self.cur_theta = self.cur_theta + self.time_step * u[1,0]
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state, axis=1)

        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)


class SingleIntegrator(Dynamics):

    def __init__(self, init_state=np.zeros((2,1))):
        """
        Single Integrator Dynamics
        
        x(t) = [x1, x2]'

        .      | 0   0 |        | 1  0 |
        x(t) = | 0   0 | x(t) + | 0  1 | u(t)
        
        y(t) = | 1  0 | x(t)
               | 0  1 |
        """
        self.params = Params()
        self.init_state = make_column(init_state)
        self.cur_state = self.init_state

        ndim = self.init_state.shape[0]
        self.A = np.zeros((ndim,ndim))
        self.B = np.identity(ndim)
        self.C = np.identity(ndim)
        
        self.time_step = self.params.step_size
        self.cur_time = 0
        self.time = np.array([0])
        self.trajectory = np.array(self.init_state)

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u=(0,0)):
        return make_column(self.B.dot(np.array(u)))

    def add_control(self, m, id):
        v = self.params.max_speed
        u = []
        for u_idx in range(len(self.cur_state)):
            u.append(m.addVar(lb=-v, ub=v, vtype=GRB.CONTINUOUS, name="agt{}_u{}".format(id, u_idx)))
        u = np.array(u)
        m.addConstr(u.transpose().dot(u) <= v**2)
        return make_column(u)

    def step(self, u):
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step

        self.cur_state = x1
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state, axis=1)

        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)


class DoubleIntegrator(Dynamics):

    def __init__(self, init_pos, init_vel=None, t0=0):
        """
        Double Integrator Dynamics
        
        x(t) = [x1, x2, v1, v2]'

        .      | 0   0   1   0 |        | 0  0 |
        x(t) = | 0   0   0   1 | x(t) + | 0  0 | u(t)
               | 0   0   0   0 |        | 1  0 |
               | 0   0   0   0 |        | 0  1 |
        
        y(t) = | 1  0  0  0 | x(t)
               | 0  1  0  0 |
        """
        self.params = Params()
        init_pos = make_column(init_pos)
        ndim = init_pos.shape[0]
        Z = np.zeros((ndim,ndim))
        I = np.identity(ndim)

        self.A = np.vstack(( np.hstack((Z, I)), np.hstack((Z, Z)) ))
        self.B = np.vstack((Z,I))
        self.C = np.hstack((I,Z))
        
        init_pos = make_column(init_pos)
        if init_vel is None:
            init_vel = np.zeros((ndim,1))

        init_state = np.vstack((init_pos, init_vel))
        self.cur_state = init_state
        self.cur_time = 0
        
        self.time = np.array([0])
        self.trajectory = np.array(init_state)
        self.time_step = self.params.step_size

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u=(0,0)):
        return self.A.dot(x) + self.B.dot(make_column(u))

    def add_control(self, m, id):
        a = self.params.max_accel
        u = []
        for u_idx in range(self.B.shape[1]):
            u.append(m.addVar(lb=-a, ub=a, vtype=GRB.CONTINUOUS, name="agt{}_u{}".format(id, u_idx)))
        u = np.array(u)
        m.addConstr(u.transpose().dot(u) <= a**2, name="agt{}_ctrlBound".format(id))
        return make_column(u)

    def step(self, u):
        u = make_column(u)
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step + 0.5*np.vstack((u,np.zeros(u.shape)))*self.time_step**2

        self.cur_state = x1
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state)

        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec
    
def repeat_control(mod, num, u=(1,0)):
    mod.print_state()

    Nstates = int(np.shape(mod.cur_state)[0])
    statsEv = np.vstack(np.zeros([Nstates, 1]))
    for i in range(num):
        mod.step(u)
        statsEv = np.append(statsEv, np.vstack(mod.cur_state), axis=1)
        print(mod)
    return statsEv, range(num)


def main(DynType):
    if DynType == 'UniEx':
        init = np.array([0, 0, 0, 0, 0, 0])
        mod = UnicycleExtended(init)
        num, u = 100, (0.1, 1)  # Number of time steps for simulation, and the controls
    if DynType == 'Uni':
        init = np.array([0, 0, 0])
        mod = Unicycle(init)
        num, u = 100, (1, .1)  # Number of time steps for simulation, and the controls

    if DynType == 'DInt':
        init = np.array([0, 0])
        mod = DoubleIntegrator(init)
        num, u = 10, (1, 0)  # Number of time steps for simulation, and the controls

    if DynType == 'SInt':
        init = np.array([0])
        mod = SingleIntegrator(init)
        num, u = 10, (0, 0)  # Number of time steps for simulation, and the controls

    statsEv, timesteps = repeat_control(mod, num, u)
    return statsEv, timesteps


if __name__ == '__main__':
    DynType = 'UniEx'  # 'Uni','SInt','DInt'
    # DynType = 'DInt'
    DynType = 'Uni'
    statsEv, timesteps = main(DynType)
    fig1 = matplotlib.pyplot
    fig1.plot(range(101), statsEv[0, :], label='q1')
    fig1.plot(range(101), statsEv[1, :], label='q2')
    fig1.plot(range(101), statsEv[2, :], label='q3')
    # fig1.plot(range(101),statsEv[3,:],label='q4')
    # fig1.plot(range(101),statsEv[4, :], label='q5')
    # fig1.plot(range(101),statsEv[5, :], label='q6')
    fig1.grid(True)
    fig1.legend()
    fig1.show()

        