import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp2", help="Directory name for loss histories", type=str)

args = parser.parse_args()

directory = args.dir

if __name__ == "__main__":
    compressions = [f.name for f in os.scandir(directory) if f.is_dir()]
    for compression in compressions:
        lambdas = [f.name for f in os.scandir(directory + '/' + compression) if f.is_dir()]
        for lmb in lambdas:
            _, _, files = next(os.walk(directory + '/' + compression + '/' + lmb))
            iterations = len(files)
            loss = 0
            for file in files:
                loss += np.load(directory + '/' + compression + '/' + lmb + '/' + file) / iterations

            plt.plot(loss)

        plt.legend(lambdas, title="lambdas")
        plt.title("Loss vs. time with " + compression)
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        plt.savefig(directory + "/Figure-" + compression + ".png")
        plt.close()
