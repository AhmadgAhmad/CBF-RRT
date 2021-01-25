"""
Safe-RRT:

Guang Yang
"""

import matplotlib.pyplot as plt
import random
import math
import copy
from simulation import Safe_Traj

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList,
                 randArea, expandDis=1, goalSampleRate=5, maxIter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start[0], start[1],0)
        self.end = Node(goal[0], goal[1],0)
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList
        self.x1_traj = [] #used to save all state along trajectory
        self.x2_traj = []
        self.theta_traj = []
        self.u_traj = [] #used to safe all control along trajectory

    def Planning(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        while True:
            # Random Sampling
            if random.randint(0, 100) > self.goalSampleRate:
                rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(
                    self.minrand, self.maxrand), random.uniform(0.0,2*math.pi)]
            else:
                rnd = [self.end.x, self.end.y, self.end.theta]

            # Find nearest node
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            # print(nind)

            # expand tree
            nearestNode = self.nodeList[nind]
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

           # Simulate Trajectory using CBF controller instead of Fixed Step Increment
            planning_obj = Safe_Traj([nearestNode.x, nearestNode.y, nearestNode.theta], rnd, self.obstacleList)
            x_simulated = planning_obj.integrate()
            self.x1_traj.extend(x_simulated[0])
            self.x2_traj.extend(x_simulated[1])
            self.theta_traj.extend(x_simulated[2])

            # Add last states of the trajectory as newNode
            newNode = copy.deepcopy(nearestNode) #initialize node structure
            newNode.x = self.x1_traj[-1]
            newNode.y = self.x2_traj[-1]
            newNode.theta = self.theta_traj[-1]
            newNode.parent = nind

            #newNode = copy.deepcopy(nearestNode)
            #newNode.x += self.expandDis * math.cos(theta)
            #newNode.y += self.expandDis * math.sin(theta)
            #newNode.parent = nind

            #if not self.__CollisionCheck(newNode, self.obstacleList):
            #    continue

            self.nodeList.append(newNode)
            print("nNodelist:", len(self.nodeList))

            # check goal
            dx = newNode.x - self.end.x
            dy = newNode.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
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

    def DrawGraph(self, rnd=None):  # pragma: no cover
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))
        return minind



class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None


def main(gx=5.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [[2,3,1.5]]  # [x,y,size]
    # Set Initial parameters
    rrt = RRT(start=[0, 0], goal=[gx, gy],
              randArea=[-2, 15], obstacleList=obstacleList)
    path = rrt.Planning(animation=show_animation)

    # Draw final path
    if show_animation:  # pragma: no cover
        rrt.DrawGraph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
