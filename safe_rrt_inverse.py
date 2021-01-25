"""
Safe-RRT:

Guang Yang
Zachary Serlin
"""

import matplotlib.pyplot as plt
import random
import math
import copy
import time
import numpy as np
from simulation_forward import Safe_Traj

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList,
                 randArea, expandDis=0.2, goalSampleRate=5, maxIter=3000):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1], start[2])
        self.goal = goal
        self.end = Node(goal[0], goal[1],0.5)
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=False):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        while True:
            feasible = True

            #random_index = random.randint(0,len(self.nodeList)-1)
            random_toward_1 = pow(random.random(), 0.5)
            random_index = int(random_toward_1*(len(self.nodeList)-1))
            # expand tree
            startNode = self.nodeList[random_index]
            desired_theta = math.atan2(self.goal[1] - startNode.y, self.goal[0] - startNode.x)

            sampled_theta = random.gauss(desired_theta, 1.0)

            startNode.theta = sampled_theta#Robot Facing Toward Sampled Position

            #Simulate Trajectory using CBF controller instead of Fixed length step Increment
            try:
                planning_obj = Safe_Traj([startNode.x, startNode.y, startNode.theta], self.obstacleList)
                x_simulated = planning_obj.integrate()

            except:
                print("QP infeasible!")
                feasible = False

            if feasible == True:
                newNode = copy.deepcopy(startNode) #initialize node structure
                newNode.xt = x_simulated[0][0:]
                newNode.yt = x_simulated[1][0:]
                newNode.theta = x_simulated[2][0:]
                newNode.parent = random_index
                newNode.x = x_simulated[0][-1]
                newNode.y = x_simulated[1][-1]
                newNode.theta = x_simulated[2][-1]
                newNode.t = 0 #needs to be parent node time + sim time\


                self.nodeList.append(newNode)


                # check goal
                dx = newNode.x - self.end.x
                dy = newNode.y - self.end.y
                d = math.sqrt(dx * dx + dy * dy)
                #if len(self.nodeList) == 200:
                #    self.DrawGraph()
                if d <= self.expandDis:
                    print("Goal!!")

                    if animation:
                        self.DrawGraph()

                    #path_x = [self.end.x]
                    #path_y = [self.end.y]


                    path_x = []
                    path_y = []


                    lastIndex = len(self.nodeList)-1
                    while self.nodeList[lastIndex].parent is not None:
                        node = self.nodeList[lastIndex]
                        #path_x.extend(node.xt)
                        #path_y.extend(node.yt)
                        #lastIndex = node.parent

                        path_x[:0]=node.xt
                        path_y[:0]=node.yt
                        lastIndex = node.parent


                    return path_x, path_y



    def DrawGraph(self):  # pragma: no cover
        """
        Draw Graph
        """

        plt.clf()

        #Plot added new Path
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.xt, node.yt, "-g")

        #Plot Obstacles
        for (x, y, size) in self.obstacleList:
            self.PlotCircle(x, y, size)


        plt.plot(self.start.x, self.start.y, "xb",markersize=30,label="Initial State")
        plt.plot(self.end.x, self.end.y, "^r",markersize=30, label="Goal State")
        plt.xlabel("$x_1$",fontsize=30)
        plt.ylabel("$x_2$",fontsize=30)
        plt.title("Safe RRT Path",fontsize=35)
        plt.xticks(size = 25)
        plt.yticks(size = 25)
        plt.axis([-2, 10, -2, 10])
        plt.grid(True)
        plt.legend( loc='top left', borderaxespad=0.)
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

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None


def main(sx=0,sy=0,gx=8, gy=8):
    print("start " + __file__)
    show_animation = True
    # ====Search Path with RRT====
    #obstacleList = [[0.3,1.2,0.20],[0.5,-0.6,0.20],[-0.5,1.3,0.20],[1.0,0.0,0.20]]  # [x,y,size]
    obstacleList = [[4.0,4.0,0.50],[6.0,6.0,0.50], [2,2,0.2]]  # [x,y,size]
    # Set Initial parameters
    start = [gx, gy,1.0]
    goal = [sx, sy]
    rrt = RRT(start=start, goal=goal,
              randArea=[-2, 10], obstacleList=obstacleList)

    print('here')
    path_x, path_y = rrt.Planning(animation=False)


    # Draw final path
    if show_animation:  # pragma: no cover
        rrt.DrawGraph()
        plt.plot(path_x, path_y, 'b-', ms=12,label='Path')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)
