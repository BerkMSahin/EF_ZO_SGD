import numpy as np
from numpy.linalg import norm


class Source:
    def __init__(self, index, beta, simulation):
        self.simulation = simulation
        self.position = np.random.uniform(low=simulation.init_size+100,
                                          high=3*simulation.init_size+100,
                                          size=(1, simulation.dim))
        self.index = index
        self.beta = beta
        self.velocity = 0

    # Determines velocity in order to actively avoid the chasing agent
    def set_velocity(self, agent):
        difference = self.position - agent.position
        self.velocity = self.beta * difference / norm(difference)

    def move(self):
        self.position += self.velocity
