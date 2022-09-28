import argparse

import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp1", help="Directory name for loss histories")

args = parser.parse_args()

directory = args.dir

if __name__ == "__main__":
    # Plot the compressions without error feedback
    compressions = [f.name for f in os.scandir(directory + "/no-error") if f.is_dir()]
    for compression in compressions:
        _, _, files = next(os.walk(directory + "/no-error/" + compression))
        iterations = len(files)
        loss = 0
        for file in files:
            loss += np.load(directory + '/no-error/' + compression + '/' + file) / iterations

        plt.title("Loss vs. time without error factor")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend(compressions)
        plt.plot(loss)

    plt.savefig(directory + "/no-error/Figure.png")
    plt.close()

    # Plot the compressions with error feedback
    compressions = [f.name for f in os.scandir(directory + "/error") if f.is_dir()]
    compressions.remove("dropout-unbiased")
    for compression in compressions:
        _, _, files = next(os.walk(directory + "/error/" + compression))
        iterations = len(files)
        loss = 0
        for file in files:
            loss += np.load(directory + '/error/' + compression + '/' + file) / iterations

        plt.title("Loss vs. time with error factor")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend(compressions)
        plt.plot(loss)

    plt.savefig(directory + "/error/Figure.png")
