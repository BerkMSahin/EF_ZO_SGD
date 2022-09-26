from simulation import Simulation
from compression import compressors as c
import numpy as np
import pandas as pd
import argparse, sys
import os

parser = argparse.ArgumentParser()
# Arguments
parser.add_argument("--eta", help="Learning rate for SGD", type=float)
parser.add_argument("--steps", help="Number of steps for SGD", type=int)
parser.add_argument("--iterations", help="Number of experiments for each case", type=int)
parser.add_argument("--N", help="Number of agents", type=int)
parser.add_argument("--R", help="Radius of agent's neighbor", type=int)
parser.add_argument("--Lambda", help="Regularization term", type=float)
parser.add_argument("--init_size", help="Board size", type=int)
parser.add_argument("--fraction_cord", help="Fraction for top-k compression", type=float)
parser.add_argument("--dropout_p", help="Dropout probability p", type=float)
parser.add_argument("--noise", help="Noise for neighboring (between 0-1)", type=float)


args = parser.parse_args()

# Creates and runs a simulation instance with specified hyperparameters

if __name__ == "__main__":

    dir = "losses"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # PARAMETERS
    ETA = args.eta #0.35
    STEPS = args.steps #10000
    ITERATIONS = args.iterations # Number of experiments for each case
    N, R, LAMBDA = args.N, args.R, args.Lambda # 20 20 5
    INIT_SIZE = args.init_size # 80
    ANIMATE = False
    FRACTION_COORDINATES = args.fraction_cord #0.5
    DROPOUT_P = args.dropout_p #0.5
    NOISE = args.noise #0.5

    compression = False
    error_factor = False

    collision_table = pd.DataFrame(data=np.zeros((ITERATIONS, 11)), columns=["Normal", c[0], c[1], c[2], c[3], c[4],
                                                                                       f"{c[0]}e", f"{c[1]}e", f"{c[2]}e", f"{c[3]}e", f"{c[4]}e"])

    for i in range(3):

        if i == 0:
            s1 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=False)

            collision_hist = s1.run(NOISE, dir)
            print("Simulation without compression has been completed.")
            collision_table["Normal"] = collision_hist
        else:
            error = (i == 2)
            # TOP-K
            s1 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[0], fraction_coordinates=FRACTION_COORDINATES,
                            error_factor=error)

            # RAND
            s2 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[1], fraction_coordinates=FRACTION_COORDINATES,
                            error_factor=error)

            # DROPOUT-BIASED
            s3 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[2], dropout_p=DROPOUT_P,
                            error_factor=error)

            # DROPOUT-UNBIASED
            s4 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[3], dropout_p=DROPOUT_P,
                            error_factor=error)
            # QSGD
            s5 = Simulation(eta=ETA, steps=STEPS,
                            iterations=ITERATIONS, n=N, r=R, Lambda=LAMBDA,
                            init_size=INIT_SIZE, animate=ANIMATE, compression=True,
                            quantization_function=c[4], num_bits=4,
                            error_factor=error)

            if error:
                tmp = "e"
            else:
                tmp = ""

            # Run the simulations
            collision_hist = s1.run(NOISE, dir)
            collision_table[c[0]+tmp] = collision_hist
            print("Simulation with TOP-K Compression was completed.")
            print("*"*40)
            collision_hist = s2.run(NOISE, dir)
            collision_table[c[1] + tmp] = collision_hist
            print("Simulation with RAND Compression was completed. ")
            print("*" * 40)
            collision_hist = s3.run(NOISE, dir)
            collision_table[c[2] + tmp] = collision_hist
            print("Simulation with DROPOUT-BIASED Compression was completed. ")
            print("*" * 40)
            collision_hist = s4.run(NOISE, dir)
            collision_table[c[3] + tmp] = collision_hist
            print("Simulation with DROPOUT-UNBIASED Compression was completed. ")
            print("*" * 40)
            collision_hist = s5.run(NOISE, dir)
            collision_table[c[4] + tmp] = collision_hist
            print("Simulation with QSGD Compression was completed. ")
            print("*" * 40)

    print(collision_table)
    collision_table.to_csv(f"./{dir}/collisions.csv") # Save the table
