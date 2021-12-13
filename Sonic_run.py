import logging

import gym
import torch
import numpy
import neat.population as pop
import neat.experiments.sonic.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
import retro
import cv2

logger = logging.getLogger(__name__)

logger.info(c.Sonic.DEVICE)
neat = pop.Population(c.Sonic)
solution, generation = neat.run()

if solution is not None:
    logger.info('Found a Solution')
    draw_net(solution, view=True, filename='./images/sonic-solution', show_disabled=True)

    # OpenAI Gym
    #env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
    env = retro.make('SonicTheHedgehog-Genesis')
    done = False
    observation = env.reset()
    ob = env.reset()
    inx, iny, inc = env.observation_space.shape
    print(inx,iny)
    inx = int(inx / 8)
    iny = int(iny / 8)

    fitness = 0
    phenotype = FeedForwardNet(solution, c.Sonic)

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(c.Sonic.DEVICE)
        ob = cv2.resize(ob, (inx, iny))
        imgarray = np.ndarray.flatten(ob)
        pred = [round(float(phenotype(imgarray)))]
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()
