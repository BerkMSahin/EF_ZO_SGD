import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--iterations", help="Number of iterations for each experiment", type=int)
parser.add_argument("--dir", default="losses", help="Directory name for loss histories")

args = parser.parse_args()

ITERATION = args.iterations
DIR = args.dir

if __name__ == "__main__":

    compressions = ["top", "rand",
                    "dropout-biased", "dropout-unbiased",
                    "qsgd"]

    # Plot No Compression Case
    no_comp_loss = 0
    for i in range(ITERATION):
        loss = np.load(f"./{DIR}/noComp{i}.npy")
        no_comp_loss += loss / ITERATION

    for i in range(2):

        if i == 0:
            error = ""
            pretitle = "Loss vs. Step (without error feedback)"
        else:
            error = "e"
            pretitle = "Loss vs. Step (with error feedback)"
            compressions.remove("dropout-unbiased")  # because it diverges

        plt.title(pretitle)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.plot(no_comp_loss)

        for cname in compressions:
            # Mean Loss
            loss_hist = 0
            for exp_idx in range(ITERATION):
                # Load the loss for exp_idx th experiment
                loss = np.load(f"./{DIR}/{cname}{error}{exp_idx}.npy")
                loss_hist += loss / ITERATION

            plt.plot(loss_hist)

        plt.legend(["NoComp"] + [cname for cname in compressions])
        plt.show()
