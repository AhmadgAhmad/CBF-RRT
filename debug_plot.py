import math
import matplotlib.pyplot as plt
import numpy as np

class Debug_plot:
    def __init__(self):
        self.CLF_traj = []

    def CLF_func(self,x1,x1s,x2,x2s,x3):
        # Convert casadi DM type to numpy array
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        self.x_norm = np.linalg.norm([x1-x1s,x2-x2s])
        CLF_margin = self.x_norm + 1 - ((x1-x1s)*math.cos(x3)/self.x_norm+(x2-x2s)*math.sin(x3)/self.x_norm)
        return CLF_margin

    def plot_CLF(self,x1_traj,x2_traj,x3_traj,delta_t,x1s,x2s):
        num_updates = len(x1_traj)
        tf = delta_t*num_updates

        for t_step in range(num_updates):
            self.CLF_margin = self.CLF_func(x1_traj[t_step],x1s,x2_traj[t_step],x2s,x3_traj[t_step])
            self.CLF_traj.append(self.CLF_margin)

        fig, ax = plt.subplots()
        t_span = np.linspace(0, tf, num=num_updates)
        ax.plot(t_span,self.CLF_traj,'b-')
        ax.set_title("CLF V(x) over Time")
        ax.set_xlabel("Time t")
        ax.set_ylabel("V")
        plt.show()
