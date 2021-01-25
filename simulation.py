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
    def __init__(self, initial_state, sampled_pos, obstacle_list):
        self.T =  100 #Total integrate time
        self.t0 = 0
        self.y0 = initial_state
        self.x_sampled = sampled_pos
        self.x_obstacle = obstacle_list # [[x1,x2,r],[x1,x2,r],[x1,x2,r],...]

        self.k_cbf = 1.2 #CBF coefficient
        self.epsilon = 0.1#Finite time CLF coefficient
        self.num_of_states = 3
        self.num_of_control_inputs = 2
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
        x1s = self.x_sampled[0]
        x2s = self.x_sampled[1]
        x3s = self.x_sampled[2]

        self.m.remove(self.m.getConstrs())

        #Lyapunov Function
        V = (-x1+x1s)**2+(x2-x2s)**2 + ((x1s-x1)/((x1-x1s)**2+(x2-x2s)**2)**0.5 + math.cos(x3))**2 + ((-x2s+x2)/((x1-x1s)**2+(x2-x2s)**2)**0.5 + math.sin(x3))**2

        # Partial derivatives for calculating time derivative
        partial_V_x1 = 2*(x1-x1s) + 2*(-(x1-x1s)*(-x1+x1s)/((x1-x1s)**2+(x2-x2s)**2)**1.5-1/((x1-x1s)**2+(x2-x2s)**2)**0.5)*((-x1+x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5+math.cos(x3))-(2*(x1-x1s)*(-x2+x2s)*((-x2+x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5)+math.sin(x3))/((x1-x1s)**2+(x2-x2s)**2)**1.5

        partial_V_x2 = 2*(x2-x2s) + 2*(-(x2-x2s)*(-x2+x2s)/((x1-x1s)**2+(x2-x2s)**2)**1.5-1/((x1-x1s)**2+(x2-x2s)**2)**0.5)*((-x2+x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5+math.sin(x3))-(2*(-x1+x1s)*(x2-x2s)*((-x1+x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5)+math.cos(x3))/((x1-x1s)**2+(x2-x2s)**2)**1.5

        partial_V_x3 = 2*math.cos(x3)*((-x2+x2s)/((x1-x1s)**2+(x2-x2s)**2)**0.5+math.sin(x3-x3s))-2*((-x1+x1s)/((x1-x1s)**2+(x2-x2s)**2)**0.5+math.cos(x3))*math.sin(x3)


        self.v = self.m.addVar(lb=self.v_lower_lim, ub=self.v_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Translation_Velocity")
        self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")
        self.delta = self.m.addVar(lb=-50, ub=50,vtype=GRB.CONTINUOUS, name="Relaxation Term")

        # Initialize Cost Function
        self.cost_func = self.v*self.v+self.w*self.w +self.delta*self.delta
        self.m.setObjective(self.cost_func, GRB.MINIMIZE)

        # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0
        for i in range(0,len(self.x_obstacle)):
            self.m.addConstr((2*(x1-self.x_obstacle[i][0])*math.cos(x3)*self.v
                                +2*(x1-self.x_obstacle[i][1])*math.sin(x3)*self.v)
                                >= -self.k_cbf*((x1-self.x_obstacle[i][0])**2+(x2-self.x_obstacle[i][1])**2-self.x_obstacle[i][2]**2),"CBF_constraint")



        #self.m.addConstr(partial_V_x1*math.cos(x3)*self.v + partial_V_x2*math.sin(x3)*self.v + partial_V_x3*self.w + self.delta <=-self.epsilon*V,"CLF_constraint")
        self.m.addConstr(partial_V_x1*math.cos(x3)*self.v + partial_V_x2*math.sin(x3)*self.v + partial_V_x3*self.w + self.delta<= -1,"CLF_constraint")


        #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        self.solution = self.m.getVars()

        # get the results of decision variables
        self.v = self.solution[0].x
        self.w = self.solution[1].x


        return [np.cos(y[2])*self.v, np.sin(y[2])*self.v, self.w]

    def integrate(self):
        t_span = np.linspace(0,self.T,num=20)
        #solution = solve_ivp(self.model,[0,self.T],self.y0, method = "RK45", t_eval = t_span, dense_output=False, events=None)
        solution = solve_ivp(fun=lambda t,y: self.model(t,y), t_span=[0,self.T], y0=self.y0, method = "RK45", t_eval = t_span, dense_output=False, events=None)
        return [solution.y[0],solution.y[1],solution.y[2]]

if __name__ == '__main__':

    initial_state = [0,0,0.5]
    obstacle_list = [[5,5,0.01]]
    sampled_pos = [-1,0,0]

    test_obj_handle = Safe_Traj(initial_state, sampled_pos, obstacle_list)
    solution = test_obj_handle.integrate()
    plt.plot(solution[0],solution[1])
    plt.show()
