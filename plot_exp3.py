import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--iterations", default=3, help="Number of iterations for each experiment", type=int)
parser.add_argument("--compression", help="Compression name for the no error case", type=str)
parser.add_argument("--compression_e", help="Compression name for the error case", type=str)
parser.add_argument("--dir", default="losses3", help="Directory name for loss histories", type=str)
parser.add_argument("--steps", help="Number of steps in the training", type=int)
parser.add_argument("--N_list", default="5 20 50 80 100", help="Number of agents you want to test", type=str)


args = parser.parse_args()

cname, cname_e = args.compression, args.compression_e

ITERATION = args.iterations
STEPS = args.steps
DIR = args.dir
listN = [float(a) for a in args.N_list.split(" ")]

if __name__ == "__main__":

    compressions = ["NoComp", cname, cname_e+"e"]

    loss_hist = np.zeros((len(compressions), len(listN), STEPS))
    counter = 0
    # Plot No Compression Case
    for filename in os.listdir("./losses3"):

        if filename == "collisions.csv":
            continue
        loss = np.load("./losses3/" + filename)
        comp_idx, n_idx = counter // (ITERATION * len(listN)), counter % len(listN)
        loss_hist[comp_idx, n_idx] += loss / ITERATION
        counter += 1
    for i, comp_name in enumerate(compressions):

        plt.title(f"{comp_name} Case")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        for j in range(len(listN)):
            plt.plot(loss_hist[i, j])
        plt.legend([f"N={n}" for n in listN])
        plt.show()
