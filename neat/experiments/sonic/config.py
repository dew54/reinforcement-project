import numpy
import torch
import gym
import retro
from neat.phenotype.feed_forward import FeedForwardNet
import cv2
import numpy as np


class Sonic:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    #71681
    NUM_INPUTS = 11200
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 100000.0

    POPULATION_SIZE = 2
    NUMBER_OF_GENERATIONS = 2
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Allow episode lengths of > than 200
    #retro.retro_env.register(
    #    id='SonicTheHedgehog-Genesis',
    #    entry_point='GreenHillZone.Act1',
    #    max_episode_steps=100000
    #)


    def fitness_fn(self, genome):
        # OpenAI Gym
        env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        #env = gym.make('SonicTheHedgehog-Genesis')
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
            #input[]
            pred = round(float(phenotype(input[:, 0:8:223, 0:8:319, :])))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

        return fitness
