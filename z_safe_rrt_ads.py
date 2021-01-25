"""
Safe-RRT:

Guang Yang
"""
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math
import copy
import csv
import time
from z_simulation_closedform_ads import MA_CBF_RRT

from cbf.obstacle import Sphere, Ellipsoid
from cbf.params import Params


show_animation = True


configPath=os.path.join(os.path.dirname(__file__),'configs.ini')
params = Params(configPath)

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, expandDis=0.3, goalSampleRate=5,u_ref_nominal=2.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        if len(start.shape) is 1:
            start = np.expand_dims(start,axis=0)
            goal = np.expand_dims(goal,axis=0)
        self.num_agents = np.size(start,0)
        self.start = Node(start[:,0], start[:,1])
        self.goal = goal
        self.end = Node(goal[:,0], goal[:,1])
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.obstacleList = obstacleList
        self.u_ref_nominal = u_ref_nominal
        self.sigma = .75

    def Planning(self, animation=False, debugPlot=0):
        """
        Pathplanning

        animation: flag for animation on or off
        debugPlot: flag to draw graph at each iteration
        """

        self.nodeList = [self.start]
        newNode = self.nodeList[0]
        while not all(newNode.atGoal):
            if debugPlot > 0 and len(self.nodeList) % debugPlot == 0:
                self.DrawGraph()

            random_index = random.randint(0,len(self.nodeList)-1)

            # expand tree
            startNode = self.nodeList[random_index]
            sampled_theta = np.zeros(self.num_agents)
            u_ref = np.zeros((self.num_agents, 2))
            for i in range(self.num_agents):
                if startNode.atGoal[i]:
                    # If this agent is already at the goal, do not need to move anymore
                    u_ref[i] = [0.0, 0.0]
                else:
                    desired_theta = math.atan2(self.goal[i][1] - startNode.y[i], self.goal[i][0] - startNode.x[i])
                    sampled_theta[i] = random.gauss(desired_theta, self.sigma)
                    # Update u_ref based on goal direction
                    u_ref[i] = [self.u_ref_nominal*math.cos(sampled_theta[i]),self.u_ref_nominal*math.sin(sampled_theta[i])]

            #Simulate Trajectory using CBF controller instead of Fixed length step Increment
            try:
                cbf_rrt_simulation = MA_CBF_RRT(np.vstack((startNode.x, startNode.y)), self.obstacleList)
                x_simulated, u_simulated= cbf_rrt_simulation.motion_planning(u_ref)
                feasible = True
            except Exception as e:
                # print(e)
                #print("QP infeasible!")
                feasible = False

            if feasible:
                newNode = copy.deepcopy(startNode) #initialize node structure
                newNode.xt = x_simulated[:self.num_agents, :]
                newNode.yt = x_simulated[self.num_agents:, :]
                newNode.ut = u_simulated

                newNode.parent = random_index
                newNode.x = x_simulated[:self.num_agents, -1]
                newNode.y = x_simulated[self.num_agents:, -1]

                T = cbf_rrt_simulation.T
                N = cbf_rrt_simulation.N
                dt = T/N
                newNode.ts = np.linspace(startNode.t+dt, startNode.t+T, N)
                # newNode.ts = np.arange(startNode.t+dt, startNode.t+1+dt/2, dt)
                newNode.t = newNode.ts[-1]
                
                

                nodeMoved = np.array(newNode.dist(startNode) > 0.01)
                # Check if you should accept the new node (min 36.50)
                if not all(np.logical_or(nodeMoved, startNode.atGoal)):
                    print('skip')
                else:
                    
                    # check goal for each agent individually
                    # Only terminate when all agents are at goal
                    for i in range(self.num_agents):
                        dx = newNode.x[i] - self.end.x[i]
                        dy = newNode.y[i] - self.end.y[i]
                        d = math.sqrt(dx * dx + dy * dy)
                        if d <= 3*self.expandDis:
                            # Node is in goal neighborhood. Check the in between states
                            dMin = self.expandDis
                            jMin = None
                            for j in range(len(newNode.ts)):
                                dx = newNode.xt[i,j] - self.end.x[i]
                                dy = newNode.yt[i,j] - self.end.y[i]
                                d = math.sqrt(dx * dx + dy * dy)
                                if d <= self.expandDis:
                                    if d < dMin:
                                        jMin = j
                                        dMin = d
                            if jMin is not None:
                                # Node is close enough. Stop at the closest point
                                newNode.atGoal[i] = True
                                newNode.xt[i,jMin:].fill(newNode.xt[i,jMin])
                                newNode.yt[i,jMin:].fill(newNode.yt[i,jMin])
                                newNode.x[i] = np.array([newNode.xt[i,-1]])
                                newNode.y[i] = np.array([newNode.yt[i,-1]])

                    # Add the new node
                    self.nodeList.append(newNode)

                    if any(newNode.atGoal):
                        print(newNode.atGoal)


        path_x = np.empty([self.num_agents,0])
        path_y = np.empty([self.num_agents,0])
        u_traj = np.zeros((2*self.num_agents,0))

        lastIndex = len(self.nodeList)-1
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            #path_x.extend(node.xt)
            #path_y.extend(node.yt)
            #lastIndex = node.parent

            path_x = np.hstack([node.xt,path_x])
            path_y = np.hstack([node.yt,path_y])
            u_traj = np.hstack([node.ut,u_traj])
            lastIndex = node.parent


        return path_x, path_y, u_traj

    def animatePaths(self, path_x, path_y, axislim=[0, 10, 0, 10]):
        plt.clf()

        colors = np.random.rand(self.num_agents,3)

        #Plot Obstacles
        for obst in self.obstacleList:
            ax = plt.gca()
            obst.plot(ax)

        # Plot start and end pos
        for i in range(self.num_agents):
            if i == 0:
                plt.plot(self.start.x[i], self.start.y[i], "xb",markersize=15,label="Initial State")
                plt.plot(self.end.x[i], self.end.y[i], "^r",markersize=15, label="Goal State")
                plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)
            else:
                plt.plot(self.start.x[i], self.start.y[i], "xb",markersize=15)
                plt.plot(self.end.x[i], self.end.y[i], "^r",markersize=15)
                plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)

        plt.xlabel("$x_1$",fontsize=15)
        plt.ylabel("$x_2$",fontsize=15)
        plt.title("CBF-RRT Path",fontsize=20)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.axis(axislim)
        plt.grid(True)
        plt.legend( loc='best', borderaxespad=0., prop={'size': 10})
        plt.rcParams.update({'font.size': 10})

        # Animate over Time
        for t in range(path_x.shape[1]):
            plt.plot(path_x[:,t], path_y[:,t], "ok", markersize=5)
            plt.pause(.05)

    def DrawGraph(self, path_x=[], path_y=[], axislim=[0, 10, 0, 10], pltZ=False):  # pragma: no cover
        """
        Draw Graph
        """

        if pltZ:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            fig,ax=plt.subplots()

        # plt.clf()

        # colors = np.random.rand(self.num_agents,3)
        colors = np.array([[0.,1.,0.],[1.,0.,1.],[0.,0.,1.],[0.,1.,0.],[1.,1.,0.],[0.,0.,0.],[1.,0.,0.]])

        #Plot added new Path
        for node in self.nodeList:
            if node.parent is not None:
                for i in range(self.num_agents):
                    if pltZ:
                        # ax.plot([node.x[i]], [node.y[i]], [node.t], "-o",  markersize=5,color=colors[i,:])
                        ax.plot(node.xt[i,:], node.yt[i,:], node.ts[:], ".-",  color=colors[i,:])
                    else:
                        parent = self.nodeList[node.parent]
                        plt.plot([node.x[i],parent.x[i]], [node.y[i],parent.y[i]], "-o", color=colors[i,:], mfc=colors[i,:], mec=[0.3,0.3,0.3])

                        # plt.plot(node.x[i], node.y[i], "-ok", markersize=5)
                        # plt.plot(node.xt[i,:], node.yt[i,:], ".-",color=colors[i,:])


        if len(path_x) > 0:
            lwidth = 8
            for i in range(self.num_agents):
                color = np.copy(colors[i,:])
                for j in range(3):
                    if color[j] == 1:
                        color[j] = 0.5
                if i == 0:
                    plt.plot(path_x[i], path_y[i], '-', linewidth=lwidth,color=color,label='Path')
                else:
                    plt.plot(path_x[i], path_y[i], '-', linewidth=lwidth, color=color)

        #Plot Obstacles
        for obst in self.obstacleList:
            ax = plt.gca()
            obst.plot(ax)

        for i in range(self.num_agents):
            if pltZ:
                pass
                # if i == 0:
                #     plt.plot([self.start.x[i]], [self.start.y[i]], [0] "xb",markersize=15,label="Initial State")
                #     plt.plot([self.end.x[i]], [self.end.y[i]], [0], "^r",markersize=15, label="Goal State")
                #     plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)
                # else:
                #     plt.plot(self.start.x[i], self.start.y[i], "xb",markersize=15)
                #     plt.plot(self.end.x[i], self.end.y[i], "^r",markersize=15)
                #     plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)
            else:
                if i == 0:
                    plt.plot(self.start.x[i], self.start.y[i], "xb",markersize=15,label="Initial State")
                    plt.plot(self.end.x[i], self.end.y[i], "^r",markersize=15, label="Goal State")
                    plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)
                else:
                    plt.plot(self.start.x[i], self.start.y[i], "xb",markersize=15)
                    plt.plot(self.end.x[i], self.end.y[i], "^r",markersize=15)
                    plt.plot(self.nodeList[-1].x[i], self.nodeList[-1].y[i], "ob", markersize=10)

        plt.xlabel("$x_1$",fontsize=15)
        plt.ylabel("$x_2$",fontsize=15)
        # plt.title("CBF-RRT Path",fontsize=20)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.axis(axislim)
        plt.grid(True)
        plt.legend( loc='best', borderaxespad=0., prop={'size': 10})
        plt.rcParams.update({'font.size': 10})

        plt.pause(.05)
        plt.show()


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
        self.atGoal = [False] * len(x)
        self.parent = None
        self.t = 0

    def dist(self, node):
        return np.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)


class World():
    """
    Used to set start, goal, and obstacles easier
    """

    def __init__(self, xlim=[-0.5,10.0], ylim=[-0.5,10.0]):
        self.start = np.empty([0,2])
        self.goal = np.empty([0,2])
        self.obstacleList = []
        self.xlim = xlim
        self.ylim = ylim

    def dist(self, x1, x2):
        return math.sqrt(x1**2 + x2**2)

    def random_agents(self, num_agents, xlim=[-5,5], ylim=[-5,5]):
        return

    def circle_agents_ellipses(self, num_agents, circle_radius=4):
        angle_diff = 2*math.pi / num_agents

        count = 0
        cur_angle = 0
        while count < num_agents:
            x = np.rint(math.cos(cur_angle) * circle_radius * 100) / 100
            y = np.rint(math.sin(cur_angle) * circle_radius * 100) / 100
            self.start = np.vstack((self.start, np.array([x+circle_radius+1,y+circle_radius+1])))
            self.goal = np.vstack((self.goal, np.array([-x+circle_radius+1,-y+circle_radius+1])))
            cur_angle += angle_diff
            count += 1

        center = (circle_radius+1,circle_radius+1)
        if num_agents == 1 or num_agents == 2 or num_agents == 4:
            self.obstacleList.append(Ellipsoid(center, (circle_radius/4, 0.1), 45))
            self.obstacleList.append(Ellipsoid(center, (circle_radius/4, 0.1), -45))
        else: # ODD
            pass


    def circle_agents(self, num_agents, circle_radius=4, agent_radius=0.5):
        angle_diff = 2*math.pi / num_agents

        count = 0
        cur_angle = 0
        while count < num_agents:
            x = np.rint(math.cos(cur_angle) * circle_radius * 100) / 100
            y = np.rint(math.sin(cur_angle) * circle_radius * 100) / 100
            self.start = np.vstack((self.start, np.array([x+circle_radius+1,y+circle_radius+1])))
            self.goal = np.vstack((self.goal, np.array([-x+circle_radius+1,-y+circle_radius+1])))
            cur_angle += angle_diff
            count += 1

    def circle_obstacles(self, num_obstacles, circle_radius=4, obst_radius=0.25):
        # place random obstacles inside the circle of agents
        for i in range(num_obstacles):
            angle = random.random() * 2 * math.pi
            pos_r = random.random() * (circle_radius - 1)
            x = math.cos(angle) * pos_r + circle_radius+1
            y = math.sin(angle) * pos_r + circle_radius+1
            obst = Sphere((x,y),0.5)
            self.obstacleList.append(obst)                

    def add_obstacle(self):
        return

    def clear_obstacles(self):
        self.obstacleList = []

    def plot_world(self):
        for i in range(len(self.start)):
            if i == 0:
                plt.plot(self.start[i,0], self.start[i,1], "xb",markersize=15,label="Initial State")
                plt.plot(self.goal[i,0], self.goal[i,1], "^r",markersize=15, label="Goal State")
            else:
                plt.plot(self.start[i,0], self.start[i,1], "xb",markersize=15)
                plt.plot(self.goal[i,0], self.goal[i,1], "^r",markersize=15)

        #Plot Obstacles
        for obst in self.obstacleList:
            ax = plt.gca()
            obst.plot(ax)

        plt.xlabel("$x_1$",fontsize=15)
        plt.ylabel("$x_2$",fontsize=15)
        plt.title("CBF-RRT Path",fontsize=20)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.axis(self.xlim + self.ylim)
        plt.grid(True)
        plt.pause(.05)
        plt.show

    def PlotCircle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-r",markersize=100)


def mc_analysis(totMcs, totAgents, loadFile=''):
    if len(loadFile) > 0:
        # Load and add to file
        checkVals = True
        times = np.load(loadFile)
        orig_totMcs, orig_totAgents = times.shape
    else:
        checkVals = False

    mcTimes = np.empty([totMcs, totAgents])
    
    mcNum = 0
    avgTime = 0
    agentNum = 0
    while agentNum < totAgents:
        print("{:d} agents avg time: {:f}".format(agentNum, avgTime))
        agentNum += 1
        w = World()
        w.circle_agents(agentNum)
        init_pos = w.start
        final_goal = w.goal
        obstacleList = w.obstacleList

        mcNum = 0
        while mcNum < totMcs:
            print("{:d} Agents, MC: {:d}".format(agentNum, mcNum))

            if checkVals and (mcNum < orig_totMcs and agentNum <= orig_totAgents):
                    mcTimes[mcNum, agentNum-1] = times[mcNum, agentNum-1]
            else:
                rrt = RRT(start=init_pos, goal=final_goal, obstacleList=obstacleList)
                startTime = time.time()
                path_x, path_y, u_traj = rrt.Planning(animation=False, debugPlot=False)
                endTime = time.time()
                mcTimes[mcNum, agentNum-1] = endTime - startTime
            mcNum += 1

        outfile = "mcData/circle_{:d}agents_{:d}mcs".format(agentNum, totMcs)
        np.save(outfile,mcTimes[:,agentNum-1])

        avgTime = np.mean(mcTimes[:,agentNum-1])

    outfile = "mcData/circle_ALL_{:d}agents_{:d}mcs".format(totAgents,totMcs)
    np.save(outfile, mcTimes)


def mc_obst_analysis(totMcs, totObst, numAgents=2, loadFile=''):
    if len(loadFile) > 0:
        # Load and add to file
        checkVals = True
        times = np.load(loadFile)
        orig_totMcs, orig_totObst = times.shape
    else:
        checkVals = False

    mcTimes = np.empty([totMcs, totObst])
    
    mcNum = 0
    avgTime = 0
    obstNum = 0
    while obstNum < totObst:
        w = World()
        w.circle_agents(numAgents)
        init_pos = w.start
        final_goal = w.goal

        mcNum = 0
        while mcNum < totMcs:
            w.clear_obstacles()
            w.circle_obstacles(obstNum)
            obstacleList = w.obstacleList

            print("{:d} Obstacles, MC: {:d}".format(obstNum, mcNum))

            if checkVals and (mcNum < orig_totMcs and obstNum <= orig_totObst):
                    mcTimes[mcNum, obstNum] = times[mcNum, obstNum]
            else:
                rrt = RRT(start=init_pos, goal=final_goal, obstacleList=obstacleList)
                startTime = time.time()
                path_x, path_y, u_traj = rrt.Planning(animation=False, debugPlot=False)
                endTime = time.time()
                mcTimes[mcNum, obstNum] = endTime - startTime
            mcNum += 1

        outfile = "mcData/obsCircle_{:d}agents_{:d}obst_{:d}mcs".format(numAgents, obstNum, totMcs)
        np.save(outfile,mcTimes[:,obstNum])

        avgTime = np.mean(mcTimes[:,obstNum])
        print("{:d} obstacles avg time: {:f}".format(obstNum, avgTime))
        obstNum += 1

    outfile = "mcData/obsCircle_ALL_{:d}agents_{:d}obst_{:d}mcs".format(numAgents,totObst,totMcs)
    np.save(outfile, mcTimes)



def main():
    print("start " + __file__)
    show_animation = True
    w = World()
    w.circle_agents_ellipses(2)
    # w.circle_agents(2)
    # w.circle_obstacles(5)
    # w.plot_world()
    init_pos = w.start
    final_goal = w.goal
    obstacleList = w.obstacleList
    rrt = RRT(start=init_pos, goal=final_goal, obstacleList=obstacleList)
    path_x, path_y, u_traj = rrt.Planning(animation=False, debugPlot=1)

    if show_animation:
        print("Num node:",len(rrt.nodeList))
        rrt.DrawGraph(path_x=path_x, path_y=path_y)
        

        rrt.animatePaths(path_x,path_y)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)
