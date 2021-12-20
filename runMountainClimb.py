import logging

import gym
import torch
import cv2
import neat.population as pop
import neat.experiments.sonic.config as c
import neat.experiments.pole_balancing.config as d
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'


print('-3')
logger = logging.getLogger(__name__)

logger.info(d.PoleBalanceConfig.DEVICE)
print('-2')
neat = pop.Population(c.Sonic)
print('0')
solution, generation = neat.run()
print('1')

if solution is not None:
    logger.info('Found a Solution')
    draw_net(solution, view=True, filename='./images/sonic', show_disabled=False)

    # OpenAI Gym
    env = gym.make('SonicTheHedgehog-Genesis')
    done = False
    observation = env.reset()
    print('2')
    fitness = 0
    phenotype = FeedForwardNet(solution, c.Sonic)

    while not done:
        env.render()
        print('3')
        input = torch.Tensor([observation]).to(c.Sonic.DEVICE)

        pred = round(float(phenotype(input)))
        print('4')
        observation, reward, done, info = env.step(pred)

        fitness += reward
        print('5')
    print('6')
    env.close()