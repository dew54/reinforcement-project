import numpy
import torch
import gym
from neat.phenotype.feed_forward import FeedForwardNet
import retro


class Sonic:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 1120
    NUM_OUTPUTS = 8
    USE_BIAS = True

    ACTIVATION = 'tanh'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 90.0

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness_fn(self, genome):
        # OpenAI Gym
        env = retro.make('SonicTheHedgehog-Genesis')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(genome, self)

        while not done:
            input = torch.Tensor(numpy.array([observation])).to(self.DEVICE)
            env.render()
            a = numpy.array([observation])
            print(a.shape)
            print(a[0, 1, 1, 1])
            # input[]
            pred = round(float(phenotype(input[:, 0:8:223, 0:8:319, :])))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

        return fitness
