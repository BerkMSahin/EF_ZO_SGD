import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp2", help="Directory name for collision histories", type=str)

args = parser.parse_args()

directory = args.dir

if __name__ == "__main__":
    compressions = [f.name for f in os.scandir(directory) if f.is_dir()]
    for compression in compressions:
        lambdas = [float(f.name) for f in os.scandir(directory + '/' + compression) if f.is_dir()]
        lambdas.sort()
        for lmb in lambdas:
            _, _, files = next(os.walk(directory + '/' + compression + '/' + str(lmb)))
            iterations = len(files)
            loss = 0
            for file in files:
                loss += np.load(directory + '/' + compression + '/' + str(lmb) + '/' + file) / iterations

            plt.plot(loss)

        plt.legend(lambdas, title="lambdas")
        plt.title("Collisions vs. time with " + compression)
        plt.xlabel("Steps")
        plt.ylabel("Collisions")

        plt.savefig(directory + "/Figure-" + compression + ".png")
        plt.close()
