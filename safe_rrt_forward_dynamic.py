"""
Safe-RRT:

Guang Yang
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import random
import math
import copy
import time
from simulation_forward_dynamic import Safe_Traj
import numpy as np

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList,
                 expandDis=0.10, goalSampleRate=5):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0],start[1],start[2],start[3],start[4],0)
        self.goal = goal
        self.T = 0.5
        self.end = Node(goal[0], goal[1],99999,999999,999999,999999) # The 999999 are dummie variables that do nothing
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.obstacleList = obstacleList
        self.v_obs_1 = -0.1
        self.v_obs_2 = 0.3
        self.path_node_list = [] #A node list contains all the nodes from the end state to the initial state
        self.nodeList = [self.start]




    def Planning(self):
        """
        Pathplanning

        animation: flag for animation on or off
        """
        #for k in range(0,20):
        while True:
            feasible = True

            random_index = random.randint(0,len(self.nodeList)-1)


            # expand tree
            startNode = self.nodeList[random_index]


            desired_theta = math.atan2(self.goal[1] - startNode.y, self.goal[0] - startNode.x)


            sampled_theta = random.gauss(desired_theta, 1.0)

            startNode.theta =  sampled_theta #Robot Facing Toward Sampled Position




            #Simulate Trajectory using CBF controller instead of Fixed length step Increment
            try:
                planning_obj = Safe_Traj([startNode.x, startNode.y, startNode.theta,startNode.x_obs, startNode.y_obs], self.obstacleList,self.T,self.v_obs_1, self.v_obs_2)

                x_simulated = planning_obj.integrate()


            except:
                print("infeasible")
                feasible = False


                #print("")

            if feasible == True:
                newNode = copy.deepcopy(startNode)
                newNode.xt = x_simulated[0][0:]
                newNode.yt = x_simulated[1][0:]
                newNode.thetat = x_simulated[2][0:]
                newNode.x_obs_t = x_simulated[3][0:]
                newNode.y_obs_t = x_simulated[4][0:]
                newNode.x_obs = x_simulated[3][-1]
                newNode.y_obs = x_simulated[4][-1]
                newNode.parent = random_index
                newNode.x = x_simulated[0][-1]
                newNode.y = x_simulated[1][-1]
                newNode.theta = x_simulated[2][-1]
                newNode.t = self.nodeList[random_index].t + self.T #needs to be parent node time + sim time
                self.nodeList.append(newNode)


                # check goal
                dx = newNode.x - self.end.x
                dy = newNode.y - self.end.y
                d = math.sqrt(dx * dx + dy * dy)
                if d <= self.expandDis:
                    print("Goal!!")

                    #if animation:
                        #self.DrawGraph()

                    lastIndex = len(self.nodeList) - 1
                    self.path_node_list = [self.nodeList[lastIndex]]

                    while self.nodeList[lastIndex].parent is not None: #Backtracking and terminate when it reaches initial node
                        node = self.nodeList[lastIndex]
                        self.path_node_list.append(node)
                        lastIndex = node.parent

                    self.state_traj = [[],[],[],[],[]]

                    for node in self.path_node_list[1:]:
                        self.state_traj[0][:0] = node.xt
                        self.state_traj[1][:0] = node.yt
                        self.state_traj[2][:0] = node.thetat
                        self.state_traj[3][:0] = node.x_obs_t
                        self.state_traj[4][:0] = node.y_obs_t


                    return self.path_node_list,  self.nodeList, self.state_traj


    #def PlotGraph(self,obstacle_list):
     #   self.fig = plt.figure()
      #  self.ax = self.fig.add_subplot(111)
       # self.robot_line, = self.ax.plot([], [], '-k')
        #self.obs_line, = self.ax.plot([], [], '--r')
        #self.obs_circle = plt.Circle((obstacle_list[0][0], obstacle_list[0][1]), obstacle_list[0][2], color='r',alpha=0.2)


    def DrawGraph(self):  # pragma: no cover
        """
        Draw Graph
        """

        plt.clf()
        plt.figure(dpi= 300)

        #Plot added new Path
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.xt, node.yt, "-g")

        #Plot Obstacles
        #for (x, y, size) in self.obstacleList:
        #    self.PlotCircle(x, y, size)

        plt.plot(self.start.x, self.start.y, "xb",markersize=10, label="Initial State")
        plt.plot(self.end.x, self.end.y, "^r",markersize=10,label="Goal State")
        plt.plot(self.state_traj[0],self.state_traj[1],'b.',label="Path")
        plt.xlabel("$x_1$",fontsize=10)
        plt.ylabel("$x_2$",fontsize=10)
        plt.title("Safe RRT with Dynamical Obstacle",fontsize=15)
        plt.legend( loc='top left', borderaxespad=0.,fontsize=10)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.axis([-1.5, 3, -1.5, 3])
        plt.grid(False)
        plt.pause(0.5)
        plt.rcParams.update({'font.size': 10})
        plt.savefig('Safe_RRT_Tree_Fig_var10.png')

    def PlotCircle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-k",markersize=5)


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, theta,x_obs,y_obs,time):
        self.x = x
        self.y = y
        self.theta = theta
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.t = time
        self.parent = None



#def main(gx=1.8, gy=2):
    #print("start " + __file__)
    # ====Search Path with RRT====
    #obstacleList = [[0.8, 0, 0.2]]  # [x,y,size]


    # Compsite Initial State start = [x1,x2,theta, x_1_obs, x_2_obs]
    #rrt_planner_obj = RRT([-0.5, -0.5, 0.5, 0.8 ,0], goal=[gx, gy], obstacleList=obstacleList)


    #path_node_list, total_node_list, state_traj = rrt_planner_obj.Planning()

    #anim_handle = rrt_planner_obj.show_animation()


    #rrt_planner_obj.DrawGraph()
    #plt.plot(state_traj[0],state_traj[1],"b",linewidth=2)
    #plt.grid(True)

    # This part is for animation


    # Draw final path


if __name__ == '__main__':
    #start_time = time.time()
    #main()

    #end_time = time.time()
    #print("Total runtime:",end_time-start_time)

    #========================================================

    obstacleList = [[0.8, 0, 0.2]]  # [x,y,size]
    goal_state = [1.8, 2.0]
    init_state = [-0.5,-0.5,0.5]


    # Compsite Initial State start = [x1,x2,theta, x_1_obs, x_2_obs]

    rrt_planner_obj = RRT([init_state[0], init_state[1], init_state[2], obstacleList[0][0] ,obstacleList[0][1]], goal=goal_state, obstacleList=obstacleList)

    start_time = time.time()
    path_node_list, total_node_list, state_traj = rrt_planner_obj.Planning()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)

    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.set_title('Safe RRT with Dynamical Obstacle',fontsize=15)

    ax.tick_params(labelsize=10)

    robot_line, = ax.plot([], [], '-k')
    obs_line, = ax.plot([], [], '--r')
    obs_circle = plt.Circle((obstacleList[0][0], obstacleList[0][1]), obstacleList[0][2], color='r',alpha=0.2)
    robot = plt.Circle((init_state[0], init_state[1]), 0.05, color='b',alpha=0.8)

    ax.plot(init_state[0], init_state[1], "xb",markersize=10,label="Initial State")
    ax.plot(goal_state[0], goal_state[1], "^r",markersize=10, label="Goal State")
    ax.legend(loc='upper right', borderaxespad=0., fontsize=10)

    def init():
        ax.set_xlim(-1.5, 3)
        ax.set_ylim(-1.5, 3)
        obs_circle.center= (obstacleList[0][0], obstacleList[0][1])
        robot.center=(init_state[0], init_state[1])
        ax.add_artist(obs_circle)
        ax.add_artist(robot)

        return robot_line,obs_line,obs_circle,robot,

    def update_data(frame):

        robot_line.set_xdata(state_traj[0][:frame])
        robot_line.set_ydata(state_traj[1][:frame])
        obs_line.set_xdata(state_traj[3][:frame])
        obs_line.set_ydata(state_traj[4][:frame])
        obs_circle.center = (state_traj[3][frame],state_traj[4][frame])
        robot.center = (state_traj[0][frame],state_traj[1][frame])

        return robot_line,obs_line,obs_circle,robot,


    anim = animation.FuncAnimation(fig, update_data, init_func=init,
                               frames=len(state_traj[0]), interval=50, blit=True, repeat=False)

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=4000)
    #anim.save('safe_RRT_dynamical_obstacle_var10.mp4', writer=writer)
    plt.show()

    rrt_planner_obj.DrawGraph()
