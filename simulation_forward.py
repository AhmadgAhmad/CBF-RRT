import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import solve_ivp
from gurobipy import *



"""
Created on Mon Feb 7 18:43:20 2019

@author: Guang Yang

The code below runs the simulation of safe-RRT Algorithm


"""
class Safe_Traj:
    def __init__(self, initial_state, obstacle_list):
        self.T =  1.5 #Total integrate time
        self.t0 = 0
        self.y0 = initial_state

        self.x_obstacle = obstacle_list # [[x1,x2,r],[x1,x2,r],[x1,x2,r],...]

        self.k_cbf_1 = 0.3 #CBF coefficient
        self.k_cbf_2 = 1.5 #CBF coefficient
        self.epsilon = 0.1#Finite time CLF coefficient
        self.num_of_states = 3
        self.num_of_control_inputs = 1
        self.v_upper_lim = 0.45 # From Create Autonomy
        self.v_lower_lim = -0.45
        self.w_upper_lim = 4.25
        self.w_lower_lim = -4.25

    def model(self, t, y):
        '''
        Integrator Notation
         y = [x1,x2,theta]
        '''
        self.m = Model("CBF_CLF_QP")
        x1 = y[0]
        x2 = y[1]
        x3 = y[2]
        self.m.remove(self.m.getConstrs())

        v = 1.0
        self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")


        # Initialize Cost Function
        self.cost_func = self.w*self.w
        self.m.setObjective(self.cost_func, GRB.MINIMIZE)

        # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0
        for i in range(0,len(self.x_obstacle)):
            h = (x1-self.x_obstacle[i][0])**2+(x2-self.x_obstacle[i][1])**2-self.x_obstacle[i][2]*self.x_obstacle[i][2]

            lfh =2*(x1-self.x_obstacle[i][0])*v*np.cos(x3) + 2*(x2-self.x_obstacle[i][1])*v*np.sin(x3)

            self.m.addConstr(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-self.x_obstacle[i][1])*np.cos(x3)-2*v*(x1-self.x_obstacle[i][0])*np.sin(x3))*self.w+self.k_cbf_1*h+self.k_cbf_2*lfh >= 0,"CBF_constraint")


        #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        self.solution = self.m.getVars()
        #self.m.write("Safe_RRT_Forward.lp")

        # get the results of decision variables
        self.w = self.solution[0].x

        #print(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-self.x_obstacle[i][1])*np.cos(x3)-2*v*(x1-self.x_obstacle[i][0])*np.sin(x3))*self.w+self.k_cbf_1*h+self.k_cbf_2*lfh)
        #print(lfh)
        #print(self.w)

        #print([x1,x2,x3])
        return [np.cos(y[2])*v, np.sin(y[2])*v, self.w]

    def integrate(self):
        t_span = np.linspace(0,self.T,num=40)
        solution = solve_ivp(fun=lambda t,y: self.model(t,y), t_span=[0,self.T], y0=self.y0, method = "RK45", t_eval = t_span, dense_output=True, events=None)
        return [solution.y[0],solution.y[1],solution.y[2]]

if __name__ == '__main__':

    initial_state = [-0.5,-0.5,1]
    #obstacle_list = [[-0.2,0.5,0.21]]
    obstacle_list = [[2,2,0.5]]
    #obstacle_list = [[3,4,1]]

    test_obj_handle = Safe_Traj(initial_state, obstacle_list)
    solution = test_obj_handle.integrate()

    fig, ax = plt.subplots()
    circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]), obstacle_list[0][2], color='r',alpha=0.2)
    ax.add_artist(circle)

    ax.plot(solution[0],solution[1])
    ax.set_xlim(-1,5)
    ax.set_ylim(-1,5)
    plt.show()
