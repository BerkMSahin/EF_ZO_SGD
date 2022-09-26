import numpy as np
from numpy.linalg import norm
from server import Server
from agent import Agent
from source import Source
from graphics import GUI
from compression import Compression
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, directory, eta=.35, alpha=.03, beta=.025, dim=2,
                 steps=10000, iterations=10, n=5, r=5,
                 Lambda=.1, init_size=100, k=1, animate=False,
                 anim_width=1600, anim_height=900, compression=False,
                 num_bits=3, quantization_function="top", dropout_p=0.5,
                 fraction_coordinates=0.5, error_factor=False, plot=False,
                 n_dropout=True, n_dropout_p=0.5, cooldown=3,
                 test_lambda=False, test_agents=False):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.steps = steps
        self.iterations = iterations
        self.n = n
        self.r = r
        self.Lambda = Lambda
        self.init_size = init_size
        self.k = k
        self.animate = animate
        self.anim_width = anim_width
        self.anim_height = anim_height
        self.losses_aggregate = []
        self.global_losses = []
        self.agents = []
        self.agent_locs = np.zeros((n, steps, dim))
        self.sources = []
        self.source_locs = np.zeros((n, steps, dim))
        self.neighbors_aggregate = {}
        self.detected_neighbors = {}
        # Compression parameters
        self.compression = compression
        self.quantization_function = quantization_function
        self.compressor = Compression(num_bits, quantization_function, dropout_p,
                                      fraction_coordinates) if compression else None
        self.error_factor = error_factor
        self.collision_counter = 0  # counts the collisions between agents
        self.plot = plot
        self.collision_hist = np.zeros((self.iterations, 1))
        self.n_dropout = n_dropout
        self.n_dropout_p = n_dropout_p
        self.directory = directory
        self.cooldown = cooldown
        self.test_lambda = test_lambda
        self.test_agents = test_agents

    @staticmethod
    def tracking_error(agent, source):
        return norm(agent.position - source.position) ** 2

    # Calculates and updates each agent's list of neighboring agents
    def calculate_neighbors(self):
        x1 = 1
        x2 = 1
        self.neighbors_aggregate.clear()
        self.detected_neighbors.clear()

        for i in range(self.n):
            self.neighbors_aggregate[i] = []
            self.detected_neighbors[i] = []

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if norm(self.agents[i].position - self.agents[j].position) < self.r:
                    self.neighbors_aggregate[i].append(self.agents[j])
                    self.neighbors_aggregate[j].append(self.agents[i])
                    if self.n_dropout:
                        x1 = np.random.rand()
                        x2 = np.random.rand()
                    if x1 > self.n_dropout_p:
                        self.detected_neighbors[i].append(self.agents[j])
                    if x2 > self.n_dropout_p:
                        self.detected_neighbors[j].append(self.agents[i])

    def count_collisions(self):
        for agent_idx in range(self.n):
            agent = self.agents[agent_idx]
            for neighbor in self.neighbors_aggregate[agent_idx]:
                # To make the dimensions consistent
                agent.position = agent.position.reshape(2, 1)
                neighbor.position = neighbor.position.reshape(2, 1)

                if abs(agent.position[0] - neighbor.position[0]) < 0.05 and abs(
                        agent.position[1] - neighbor.position[1]) < 0.05 and agent.cooldown == 0\
                        and neighbor.cooldown == 0:
                    self.collision_counter += 1
                    agent.cooldown = self.cooldown
                    neighbor.cooldown = self.cooldown

    def run(self):
        for i in range(self.iterations):
            np.random.seed(i + 5)  # set random seed
            self.collision_counter = 0
            server = Server(self)
            for j in range(self.n):
                agent = Agent(j, self)
                self.agents.append(agent)

                source = Source(j, self.beta, self)
                self.sources.append(source)

            for j in range(self.steps):
                if j % self.k == 0:
                    self.calculate_neighbors()
                self.losses_aggregate.append([])

                for k in range(self.n):
                    agent = self.agents[k]
                    source = self.sources[k]

                    if agent.cooldown != 0:
                        agent.cooldown -= 1

                    agent.compute_grad(source)

                self.count_collisions()

                for k in range(self.n):
                    agent = self.agents[k]
                    local_grad = agent.local_grad
                    if self.compressor is not None:
                        if self.error_factor:
                            local_grad_e = local_grad + agent.error
                            local_grad = self.compressor.quantize(local_grad_e.T).T
                            agent.error = local_grad_e - local_grad
                        else:
                            local_grad = self.compressor.quantize(local_grad.T).T
                    server.local_grads.append(local_grad)

                server.aggregate()

                if self.animate and i + 1 == self.iterations:
                    for k in range(self.n):
                        self.agent_locs[k][j] = self.agents[k].position
                        self.source_locs[k][j] = self.sources[k].position

                for k in range(self.n):
                    agent = self.agents[k]
                    source = self.sources[k]

                    source.set_velocity(agent)
                    source.move()

                    error = self.tracking_error(agent, source)
                    self.losses_aggregate[j].append(error)

                self.losses_aggregate[j] = np.array(self.losses_aggregate[j])

            global_loss = np.mean(np.array(self.losses_aggregate), axis=1)
            self.losses_aggregate = []
            self.global_losses.append(global_loss)

            # Save the losses
            if self.compression:
                if self.error_factor:
                    file_name = f"./{self.directory}/{self.quantization_function}e{i}"
                else:
                    file_name = f"./{self.directory}/{self.quantization_function}{i}"
            else:
                file_name = f"./{self.directory}/noComp{i}"

            if self.test_lambda:
                file_name += f"lamb{round(self.Lambda)}"

            if self.test_agents:
                file_name += f"N{self.n}"

            np.save(file_name, global_loss)  # save the loss history
            self.collision_hist[i] = self.collision_counter  # save the collision count
            print(f"Experiment {i} has been completed.")

        final_loss = np.mean(np.array(self.global_losses), axis=0)
        print("Final loss:", final_loss[-1])
        print(f"Number of collisions: {self.collision_counter}")

        if self.animate:
            gui = GUI(self.anim_width, self.anim_height, self)
            gui.animate(self.agent_locs, self.source_locs)

            np.save(f"./{self.directory}/loss_hist.npy", final_loss)

        if self.plot:
            plt.xlabel('Steps')
            plt.ylabel('Loss value')
            plt.title('Zeroth order federated tracking, n=' + str(self.n) +
                      ", r=" + str(self.r) +
                      ", lambda=" + str(self.Lambda) + ",\n" +
                      "iterations=" + str(self.iterations) +
                      ", final loss=" + str(final_loss[-1]) +
                      ", Compression: " + self.quantization_function)
            plt.plot(final_loss)
            plt.grid(which="major")
            plt.show()
        return self.collision_hist
