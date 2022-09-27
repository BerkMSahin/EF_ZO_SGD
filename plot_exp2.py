import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--iterations", default=3, help="Number of iterations for each experiment", type=int)
parser.add_argument("--compression", help="Compression name for the no error case", type=str)
parser.add_argument("--compression_e", help="Compression name for the error case", type=str)
parser.add_argument("--dir", default="losses2", help="Directory name for loss histories", type=str)
parser.add_argument("--steps", help="Number of steps in the training", type=int)
parser.add_argument("--lambda_list", default="1 2.5 5 7 10", help="Number of lambdas you want to test", type=str)


args = parser.parse_args()

cname, cname_e = args.compression, args.compression_e

ITERATION = args.iterations
STEPS = args.steps
DIR = args.dir
lambda_list = [float(a) for a in args.lambda_list.split(" ")]

if __name__ == "__main__":

    compressions = ["NoComp", cname, cname_e+"e"]

    loss_hist = np.zeros((len(compressions), len(lambda_list), STEPS))
    counter = 0
    # Plot No Compression Case
    for filename in os.listdir("./losses2"):
        if filename == "collisions.csv":
            continue
        loss = np.load("./losses2/" + filename)
        comp_idx, lamb_idx = counter // (ITERATION * len(lambda_list)), counter % len(lambda_list)
        loss_hist[comp_idx, lamb_idx] += loss / ITERATION
        counter += 1
    for i, comp_name in enumerate(compressions):

        plt.title(f"{comp_name} Case")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        for j, lamb in enumerate(lambda_list):
            plt.plot(loss_hist[i, j])
        plt.legend([f"lambda={lmb}" for lmb in lambda_list])
        plt.show()
