"""
Safe-RRT:

Guang Yang
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
import csv
import time
from simulation_closedform import CBF_RRT


show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, expandDis=0.45, goalSampleRate=5,u_ref_nominal=2.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.goal = goal
        self.end = Node(goal[0], goal[1])
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.obstacleList = obstacleList
        self.u_ref_nominal = u_ref_nominal

    def Planning(self, animation=False):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        while True:

            random_index = random.randint(0,len(self.nodeList)-1)

            # expand tree
            startNode = self.nodeList[random_index]
            desired_theta = math.atan2(self.goal[1] - startNode.y, self.goal[0] - startNode.x)

            sampled_theta = random.gauss(desired_theta, 0.3)

            # Update u_ref based on goal direction
            u_ref = [self.u_ref_nominal*math.cos(sampled_theta),self.u_ref_nominal*math.sin(sampled_theta)]

            #print(startNode.x,startNode.y)

            #Simulate Trajectory using CBF controller instead of Fixed length step Increment
            try:
                cbf_rrt_simulation = CBF_RRT(np.array([[startNode.x],[startNode.y]]), self.obstacleList)
                x_simulated, u_simulated= cbf_rrt_simulation.motion_planning(u_ref)
                feasible = True

            except:
                #print("QP infeasible!")
                feasible = False

            if feasible == True:
                newNode = copy.deepcopy(startNode) #initialize node structure
                newNode.xt = x_simulated[0][0:]
                newNode.yt = x_simulated[1][0:]
                newNode.ut = u_simulated[0:]

                newNode.parent = random_index
                newNode.x = x_simulated[0][-1]
                newNode.y = x_simulated[1][-1]

                newNode.t = 0 #needs to be parent node time + sim time\

                self.nodeList.append(newNode)


                # check goal
                dx = newNode.x - self.end.x
                dy = newNode.y - self.end.y
                d = math.sqrt(dx * dx + dy * dy)
                if d <= self.expandDis:
                    print("Goal!!")

                    if animation:
                        self.DrawGraph()

                    path_x = []
                    path_y = []
                    u_traj = np.zeros((2,0))


                    lastIndex = len(self.nodeList)-1
                    while self.nodeList[lastIndex].parent is not None:
                        node = self.nodeList[lastIndex]
                        #path_x.extend(node.xt)
                        #path_y.extend(node.yt)
                        #lastIndex = node.parent

                        path_x[:0]=node.xt
                        path_y[:0]=node.yt
                        #u_traj[:0]= node.ut
                        u_traj = np.hstack((u_traj, node.ut))
                        lastIndex = node.parent


                    return path_x, path_y, u_traj



    def DrawGraph(self):  # pragma: no cover
        """
        Draw Graph
        """

        plt.clf()

        #Plot added new Path
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.x, node.y, "-ok",markersize=5)
                plt.plot(node.xt, node.yt, "-g.")

        #Plot Obstacles
        for (x, y, size) in self.obstacleList:
            self.PlotCircle(x, y, size)


        plt.plot(self.start.x, self.start.y, "xb",markersize=15,label="Initial State")
        plt.plot(self.end.x, self.end.y, "^r",markersize=15, label="Goal State")
        plt.xlabel("$x_1$",fontsize=15)
        plt.ylabel("$x_2$",fontsize=15)
        plt.title("CBF-RRT Path",fontsize=20)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.axis([-1.5, 8, -1.5, 8])
        plt.grid(True)
        plt.legend( loc='best', borderaxespad=0., prop={'size': 20})
        plt.pause(0.5)
        plt.rcParams.update({'font.size': 30})

    def PlotCircle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-r",markersize=100)


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def main():
    print("start " + __file__)
    show_animation = True
    # ====Search Path with RRT====
    #obstacleList = [[0.3,1.2,0.20],[0.5,-0.6,0.20],[-0.5,1.3,0.20],[1.0,0.0,0.20]]  # [x,y,size]
    obstacleList = [[-0.0,1.0,0.50],[1.7,-0.5,0.5],[3,3,0.5],[3,2.0,0.5],[2,5,0.5]]  # [x,y,size]
    final_goal = [5.0,5.0]
    # Set Initial parameters

    rrt = RRT(start=[-0.5, -0.5], goal=final_goal, obstacleList=obstacleList)


    path_x, path_y, u_traj = rrt.Planning(animation=False)


    # Draw final path
    if show_animation:  # pragma: no cover
        rrt.DrawGraph()
        plt.plot(path_x, path_y, 'b-', ms=12,label='Path')
        plt.grid(True)
        plt.show()

    np.save("u_traj", u_traj)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)
