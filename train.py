import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math


#%matplotlib inline

import sys
sys.path.append('../../')

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame


class Train: 
    def __init__(self, args ):
        
        
        state = 'GreenHillZone.Act1'
        self.args = args

        self.env = retro.make(game='SonicTheHedgehog-Genesis', state = state, scenario='contest')
        self.env.seed(0)
        torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)

        self.possible_actions = {
                    # No Operation
                    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    # Left
                    1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    # Right
                    2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    # Left, Down
                    3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    # Right, Down
                    4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                    # Down
                    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    # Down, B
                    6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    # B
                    7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                }

        INPUT_SHAPE = (4, 84, 84)
        ACTION_SIZE = len(self.possible_actions)
        SEED = 0
        GAMMA = 0.99           # discount factor
        BUFFER_SIZE = 100000   # replay buffer size
        BATCH_SIZE = 32        # Update batch size
        LR = 0.0001            # learning rate 
        TAU = 1e-3             # for soft update of target parameters
        UPDATE_EVERY = 100     # how often to update the network
        UPDATE_TARGET = 10000  # After which thershold replay to be started 
        EPS_START = 0.99       # starting value of epsilon
        EPS_END = 0.01         # Ending value of epsilon
        EPS_DECAY = 100         # Rate by which epsilon to be decayed

        if args["useNEAT"]:
            pass
        else:
            pass

        self.agent = None
        self.agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn, '', args)

        
        self.start_epoch = 0
        self.scores = []
        self.scores_window = deque(maxlen=20)


        self.epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
        plt.plot([self.epsilon_by_epsiode(i) for i in range(1000)])
        plt.savefig('epsilon_on_episodes')
        state = self.stack_frames(None, self.env.reset(), True) 





    def stack_frames(self, frames, state, is_new=False):
        frame = preprocess_frame(state, (1, -1, -1, 1), 84)
        frames = stack_frame(frames, frame, is_new)

        return frames    

    def train(self, n_episodes=1000):
        if self.args["useDDQN"]:
            print("Double DQN")
        else:
            print("Vanilla DQN")
        """
        Params
        ======
            n_episodes (int): maximum number of training episodes
        """
        for i_episode in range(self.start_epoch + 1, n_episodes+1):
            state = self.stack_frames(None, self.env.reset(), True)
            self.score = 0
            eps = self.epsilon_by_epsiode(i_episode)

            # Punish the agent for not moving forward
            prev_state = {}
            steps_stuck = 0
            timestamp = 0

            while timestamp < 1000:
                action = self.agent.act(state, eps)
                next_state, reward, done, info = self.env.step(self.possible_actions[action])
                self.score += reward

                timestamp += 1

                # Punish the agent for standing still for too long.
                if (prev_state == info):
                    steps_stuck += 1
                else:
                    steps_stuck = 0
                prev_state = info
        
                if (steps_stuck > 20):
                    reward -= 1
                
                next_state = self.stack_frames(state, next_state, False)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            self.scores_window.append(self.score)       # save most recent score
            self.scores.append(self.score)              # save most recent score


            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(self.scores_window), eps))

        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        if self.args["useDDQN"]:
            plt.savefig('scores_on_episodes_DDQN')
        else:
            plt.savefig('scores_on_episodes_dqn')
        print("Trained!")   

        return self.scores
        
