import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="exp3", help="Directory name for loss histories", type=str)

args = parser.parse_args()

directory = args.dir

if __name__ == "__main__":
    compressions = [f.name for f in os.scandir(directory) if f.is_dir()]
    for compression in compressions:
        Ns = [int(f.name) for f in os.scandir(directory + '/' + compression) if f.is_dir()]
        Ns.sort()
        for N in Ns:
            _, _, files = next(os.walk(directory + '/' + compression + '/' + str(N)))
            iterations = len(files)
            loss = 0
            for file in files:
                loss += np.load(directory + '/' + compression + '/' + str(N) + '/' + file) / iterations

            plt.plot(loss)

        plt.legend(Ns, title="Agents")
        plt.title("Loss vs. time with " + compression)
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        plt.savefig(directory + "/Figure-" + compression + ".png")
        plt.close()
