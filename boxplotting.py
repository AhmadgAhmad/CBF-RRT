import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

def main():
    
    times = np.load("mcData/circle_ALL_10agents_100mcs.npy")

    data = [times[:,i] for i in range(times.shape[1])]
    fig, ax = plt.subplots()
    ax.boxplot(data, showfliers=False)

    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)