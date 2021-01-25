"""
Safe Rapidly-Exploring Random Trees (Safe-RRT)

Author: Zachary Serlin (zserlin@bu.edu)
Feb 4 2018
"""

import matplotlib.pyplot as plt
import random
import math
import copy
import numpy as np
import time

from cbf import QPcontroller
from robot import iRobot
from progress import steer

show_animation = False

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, expandt=1.0, goalSampleRate=5, maxIter=500,num_agents=1,past_paths=None):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Samping Area [min,max]
        """

        self.start = Node(start[0],start[1],t=0)
        self.end = Node(goal[0],goal[1])
        self.bounds = randArea
        self.expandt = expandt
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList #[[x,y,xdot,ydot],[x,y,xdot,ydot]]
        self.num_agents = num_agents
        self.past_paths = past_paths
        self.sigma = 10

    def Planning(self,safe_size,agent_size,animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        while True:

            # Random Sampling
            if random.randint(0, 100) > self.goalSampleRate:
                #if restarted == 0:
                #    rnd = [np.random.normal(self.end.x, self.sigma), np.random.normal(
                #        self.end.y, self.sigma)]
                rnd = [random.uniform(self.bounds[0], self.bounds[1]), random.uniform(
                    self.bounds[2], self.bounds[3])]
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            # print(nind)

            # expand tree
            nearestNode = self.nodeList[nind]

            newNode = copy.deepcopy(nearestNode)
            newNode.parent = nind

            end,control,tim = steer(nearestNode,rand,obstacles,end_time,delta_t)



            if distance(nearestNode,end) < distance_thres:
                continue

            newNode.x = end[0]
            newNode.y = end[1]
            newNode.controlv = cbf_controlv
            newNode.controlw = cbf_controlw
            newNode.t += 0 #Put cbf steer time here

            self.nodeList.append(newNode)

            # check goal
            dx = newNode.x - self.end.x
            dy = newNode.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandt:
                print("Goal!!")
                break

            if animation:
                self.DrawGraph(rnd)

        path = [[self.end.x, self.end.y]]
        lastIndex = len(self.nodeList) - 1
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def DrawGraph(self,bounds, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")
        for (x, y, size) in self.obstacleList:
            self.PlotCircle(x, y, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis(bounds)
        plt.grid(True)
        plt.pause(0.01)

    def PlotCircle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-k")

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))
        return minind

class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, t=0):
        self.x = x
        self.y = y
        self.parent = None
        self.t = t


def main():
    # ====Search Path with RRT====
    # Parameter

    t = time.time()
    '''
    obstacleList = [
        [45, 15, 7],
        [5, 35, 7],
        [25, 65, 7],
        [20, 50, 7],
        [40, 50, 7],
        [25, 75, 7],
        [35, 105, 7],
        [15, 115, 7],
        [45, 140, 14],
        [15, 160, 7],
        [45, 160, 7],
    ]  # [x,y,size]
    '''
    obstacleList = [
        [15,45,7],
        [35,5,7],
        [65,25,7],
        [50,20,7],
        [50,40,7],
        [75,25,7],
        [105,35,7],
        [115,15,7],
        [140,15,14],
        [160,15,7],
        [160,45,7],
    ]  # [x,y,size]
    past_paths = []
    start=[(5, 5),(30,20)]
    goal=[(125,30),(165,30)]
    num_agents = len(start)
    bounds = [0, 170, 0, 60]
    plot_bounds = [0, 170, 0, 60]
    agent_radius = [7,7]
    for i in range(0,num_agents):

        rrt = RRT(start=start[i], goal=goal[i],
                  randArea=bounds, obstacleList=obstacleList,past_paths=past_paths)
        #rrt = RRT(start=[(0, 0),(10,0)], goal=[(5, 10),(0,10)],
        #          randArea=[-2, 15], obstacleList=obstacleList)
        path = rrt.Planning(safe_size=agent_radius,agent_size=agent_radius[i],animation=False)
        past_paths.append((path[::-1]))
        # Path smoothing
        #maxIter = 1000
        #smoothedPath = PathSmoothing(path, maxIter, obstacleList)
    print('All Found')
    print('Time:',(time.time()-t))
    # Draw final path
    if show_animation:
        #rrt.DrawGraph(plot_bounds)
        for i in range(0,num_agents):
            plt.plot([x for (x, y) in past_paths[i]], [y for (x, y) in past_paths[i]], '-r')
            #plt.plot([x for (x, y) in smoothedPath], [
            #    y for (x, y) in smoothedPath], '-b')
            plt.plot(start[i][0], start[i][1], "or")
            plt.plot(goal[i][0], goal[i][1], "xr")
        #plt.xlim(plot_bounds[0],plot_bounds[1])
        #plt.ylim(plot_bounds[2],plot_bounds[3])
        plt.axis(plot_bounds)
        plt.grid(True)
        for (x, y, size) in obstacleList:
            PlotCircle(x, y, size,plot_bounds)
        plt.pause(0.01)
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()

    #Play Trajectories as video
    max_time = 0
    for i in range(0,num_agents):
        if len(past_paths[i]) > max_time:
            max_time = len(past_paths[i])
    for t in range(0,max_time):
        DrawGraph(plot_bounds,past_paths,num_agents,start,goal,agent_radius,obstacleList,t)
        plt.pause(.01)
        time.sleep(.01)

def DrawGraph(bounds,past_paths,num_agents,start,goal,agent_radius,obstacleList,t):
    plt.clf()
    for j in range(0,num_agents):
        max_time = len(past_paths[j])
        if max_time > t:
            plt.plot(start[j][0], start[j][1], ".r")
            plt.plot(goal[j][0], goal[j][1], "xr")
            plt.plot(past_paths[j][t][0],past_paths[j][t][1],"sb")
        else:
            plt.plot(start[j][0], start[j][1], ".r")
            plt.plot(goal[j][0], goal[j][1], "xr")
            plt.plot(goal[j][0], goal[j][1],"sb")

    for (x, y, size) in obstacleList:
        PlotCircle(x, y, size,bounds)
    #plt.xlim(bounds[0],bounds[1])
    #plt.ylim(bounds[2],bounds[3])
    plt.axis(bounds)
    plt.grid(True)

def PlotCircle(x, y, size,bounds):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(math.radians(d)) for d in deg]
    yl = [y + size * math.sin(math.radians(d)) for d in deg]
    #plt.xlim(bounds[0],bounds[1])
    #plt.ylim(bounds[2],bounds[3])
    plt.axis(bounds)
    plt.grid(True)
    plt.plot(xl, yl, "-k")

if __name__ == '__main__':
    main()
