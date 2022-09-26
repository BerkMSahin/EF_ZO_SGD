import numpy as np
from numpy.linalg import norm


class Agent:
    def __init__(self, index, simulation):
        self.simulation = simulation
        self.position = np.random.uniform(low=-simulation.init_size,
                                          high=simulation.init_size,
                                          size=(1, simulation.dim))
        self.index = index
        self.velocity = 0
        self.local_grad = np.zeros((simulation.n, simulation.dim))
        self.error = np.zeros((simulation.n, simulation.dim))

    # Loss functions for gradient estimation, with respect to the pursued source
    def loss(self, source):
        return .5 * (norm(self.position - source.position) ** 2)

    def loss_plus(self, source, u):
        return .5 * (norm(self.position + self.simulation.eta * u - (source.position + .5 * source.velocity)) ** 2)

    # Loss functions for gradient estimation, with respect to neighboring agents
    # (used for the regularization terms)
    def loss_reg(self, neighbor):
        return self.simulation.Lambda * (norm(self.position - neighbor.position)**2 - self.simulation.r**2)

    def loss_reg_plus(self, neighbor, u):
        return self.simulation.Lambda * (norm(self.position +
                                              self.simulation.eta * u -
                                              (neighbor.position + .5 * neighbor.velocity))**2 -
                                         self.simulation.r**2)

    # Computes the local gradient of the agent
    def compute_grad(self, source):
        u = np.random.standard_normal((1, self.simulation.dim))
        local_grad = np.zeros((self.simulation.n, self.simulation.dim))
        grad_i = u * (self.loss_plus(source, u) - self.loss(source)) / self.simulation.eta
        local_grad[self.index, :] = grad_i
        for neighbor in self.simulation.detected_neighbors[self.index]:
            u = np.random.standard_normal((1, self.simulation.dim))
            grad_j = u * (self.loss_reg_plus(neighbor, u) - self.loss_reg(neighbor)) / self.simulation.eta
            local_grad[neighbor.index, :] = -grad_j

        self.local_grad = local_grad
