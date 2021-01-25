import matplotlib.pyplot as plt
from matplotlib import animation
import math
import numpy as np
from scipy.integrate import solve_ivp
from gurobipy import *
from matplotlib import rc



"""
Created on Mon Feb 7 18:43:20 2019

@author: Guang Yang

The code below runs the simulation of safe-RRT Algorithm


"""
class Safe_Traj:
    def __init__(self, initial_state, obstacle_list,T,v_obs_1,v_obs_2):
        self.T =  T #Total integrate time
        self.t0 = 0
        self.y0 = initial_state

        #self.x_sampled = sampled_pos
        self.x_obstacle = obstacle_list # [[x1,x2,r],[x1,x2,r],[x1,x2,r],...]
        #0.4 2.5
        self.k_cbf_1 = 1.0 #CBF coefficient
        self.k_cbf_2 = 1.9 #CBF coefficient

        #self.k_cbf_1 = 1.5 #CBF coefficient
        #self.k_cbf_2 = 1.8#CBF coefficient
        #self.epsilon = 0.1#Finite time CLF coefficient
        self.num_of_states = 3
        self.num_of_control_inputs = 1
        self.v_upper_lim = 0.45 # From Create Autonomy
        self.v_lower_lim = -0.45
        self.w_upper_lim = 4.25
        self.w_lower_lim = -4.25
        self.v_obs_1 = v_obs_1  #V_x
        self.v_obs_2 = v_obs_2  #V_y

        self.v_obs_1_op = -v_obs_1  #V_x
        self.v_obs_2_op = -v_obs_2  #V_y



    def model(self, t, y):
        '''
        Integrator Notation
         y = [x1,x2,theta]
        '''
        self.m = Model("CBF_CLF_QP")
        x1 = y[0]
        x2 = y[1]
        x3 = y[2]

        x_obs_1 = y[3]
        x_obs_2 = y[4]

        x_obs_2_1 = y[5]
        x_obs_2_2 = y[6]

        x_obs_3_1 = y[7]
        x_obs_3_2 = y[8]

        x_obs_4_1 = y[9]
        x_obs_4_2 = y[10]
        self.m.remove(self.m.getConstrs())

        v = 1.0
        self.w = self.m.addVar(lb=self.w_lower_lim, ub=self.w_upper_lim,vtype=GRB.CONTINUOUS, name="Control_Angular_Velocity")


        # Initialize Cost Function
        self.cost_func = self.w*self.w
        self.m.setObjective(self.cost_func, GRB.MINIMIZE)

        # CBF Constraint for h(x) = (x1 + x_{obs,1})^2 + (x2 + x_{obs,2})^2 - r^2>= 0


        # Obstacle 1 Constraint
        #h = (x1-x_obs_1)**2+(x2-x_obs_2)**2-self.x_obstacle[0][2]*self.x_obstacle[0][2]
        h = (x1-x_obs_1)**2+(x2-x_obs_2)**2-self.x_obstacle[0][2]*self.x_obstacle[0][2]

        lfh =2*(x1-x_obs_1)*v*np.cos(x3) + 2*(x2-x_obs_2)*v*np.sin(x3)-2*(x1-x_obs_1)*self.v_obs_1 - 2*(x2-x_obs_2)*self.v_obs_2

        self.m.addConstr(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-x_obs_2)*np.cos(x3) -2*v*(x1-x_obs_1)*np.sin(x3))*self.w+ 2*self.v_obs_1**2 + 2*self.v_obs_2**2+self.k_cbf_1*h+self.k_cbf_2*lfh >= 0,"CBF_constraint")

        #Obstacle 2 Constraint
        h_2 = (x1-x_obs_2_1)**2+(x2-x_obs_2_2)**2-self.x_obstacle[1][2]*self.x_obstacle[1][2]

        lfh_2 =2*(x1-x_obs_2_1)*v*np.cos(x3) + 2*(x2-x_obs_2_2)*v*np.sin(x3)-2*(x1-x_obs_2_1)*self.v_obs_1 - 2*(x2-x_obs_2_2)*self.v_obs_2

        self.m.addConstr(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-x_obs_2_2)*np.cos(x3) -2*v*(x1-x_obs_2_1)*np.sin(x3))*self.w+ 2*self.v_obs_1**2 + 2*self.v_obs_2**2+self.k_cbf_1*h_2+self.k_cbf_2*lfh_2 >= 0,"CBF_constraint_2")

        #Obstacle 3 Constraint
        h_3 = (x1-x_obs_3_1)**2+(x2-x_obs_3_2)**2-self.x_obstacle[2][2]*self.x_obstacle[2][2]

        lfh_3 =2*(x1-x_obs_3_1)*v*np.cos(x3) + 2*(x2-x_obs_3_2)*v*np.sin(x3)-2*(x1-x_obs_3_1)*self.v_obs_1 - 2*(x2-x_obs_3_2)*self.v_obs_2

        self.m.addConstr(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-x_obs_3_2)*np.cos(x3) -2*v*(x1-x_obs_3_1)*np.sin(x3))*self.w+ 2*self.v_obs_1_op**2 + 2*self.v_obs_2_op**2+self.k_cbf_1*h_3+self.k_cbf_2*lfh_3 >= 0,"CBF_constraint_3")

        #Obstacle 4 Constraint
        h_4 = (x1-x_obs_4_1)**2+(x2-x_obs_4_2)**2-self.x_obstacle[3][2]*self.x_obstacle[3][2]

        lfh_4 =2*(x1-x_obs_4_1)*v*np.cos(x3) + 2*(x2-x_obs_4_2)*v*np.sin(x3)-2*(x1-x_obs_4_1)*self.v_obs_1 - 2*(x2-x_obs_4_2)*self.v_obs_2

        self.m.addConstr(2*x1*v*np.cos(x3)*v*np.cos(x3) + 2*x2*v*np.sin(x3)*v*np.sin(x3)+(2*v*(x2-x_obs_4_2)*np.cos(x3) -2*v*(x1-x_obs_4_1)*np.sin(x3))*self.w+ 2*self.v_obs_1_op**2 + 2*self.v_obs_2_op**2+self.k_cbf_1*h_4+self.k_cbf_2*lfh_4 >= 0,"CBF_constraint_4")

        #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        self.solution = self.m.getVars()


        #self.m.write("Safe_RRT_Forward.lp")

        # get the results of decision variables
        self.w = self.solution[0].x

        return [np.cos(y[2])*v, np.sin(y[2])*v, self.w, self.v_obs_1, self.v_obs_2,self.v_obs_1, self.v_obs_2,self.v_obs_1_op, self.v_obs_2_op,self.v_obs_1_op, self.v_obs_2_op]

    def integrate(self):
        t_span = np.linspace(0,self.T,num=30)

        solution = solve_ivp(fun=lambda t,y: self.model(t,y), t_span=[0,self.T], y0=self.y0, method = "RK45", t_eval = t_span, dense_output=True, events=None)
        print(solution)
        return [solution.y[0],solution.y[1],solution.y[2],solution.y[3],solution.y[4],solution.y[5],solution.y[6],solution.y[7],solution.y[8],solution.y[9],solution.y[10]]
        #return [solution.y[0],solution.y[1],solution.y[2],solution.y[3],solution.y[4]]

    def PlotCircle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-k",markersize=5)




if __name__ == '__main__':


    obstacle_list = [[0.5,0,0.2]]



    test_obj_handle = Safe_Traj([-0.5,-0.5,0.8,0.5,0],obstacle_list,0.1,-0.1,0.1)
    solution = test_obj_handle.integrate()

    # Plot Circle
    #ax.add_artist(circle)

    #ax.plot(solution[0],solution[1])
    #ax.plot(solution[3],solution[4])
    #ax.set_xlim(-1,5)
    #ax.set_ylim(-1,5)

    solution = np.asanyarray(solution)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    robot_line, = ax.plot([], [], '-k')
    obs_line, = ax.plot([], [], '--r')
    obs_circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]), obstacle_list[0][2], color='r',alpha=0.2)

    def init():
        ax.set_xlim(-1.5, 3)
        ax.set_ylim(-1.5, 3)
        obs_circle.center= (obstacle_list[0][0], obstacle_list[0][1])
        ax.add_artist(obs_circle)

        return robot_line,obs_line,obs_circle,

    def update_data(frame):

        robot_line.set_xdata(solution[0][:frame])
        robot_line.set_ydata(solution[1][:frame])
        obs_line.set_xdata(solution[3][:frame])
        obs_line.set_ydata(solution[4][:frame])
        obs_circle.center = (solution[3][frame],solution[4][frame])

        return robot_line,obs_line,obs_circle,


    anim = animation.FuncAnimation(fig, update_data, init_func=init,
                               frames=len(solution[0]), interval=50, blit=True, repeat=False)

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=2000)
    #anim.save('lines.mp4', writer=writer)
    plt.show()
