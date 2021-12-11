# reinforcement-project
RL academic project

Note: 
Retro-gym is not compatible with python 3.9 or higher. Install python 3.7, then for installing packages for a specific version of python:
py -3.7 -m pip install gym-retro

Import a rom:
py -3.7 -m retro.import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'

per il training della nn, una registrazione di gioco di uno di noi sarà usata per il training e quella dell'altro per la validation

TODO
1) write pressed keys in csv
2) train NN with that csv
3) import NN structure and weights to feed NEAT
4) transfer learning on new levels
5) explain maths

We'll focus on training an agent to play an old style video game ('Sonic the hedgehog' by 'SEGA'), implementing a Deep-Q learning algorithm.
We've found an open GitHub repository that implements a simple DQN; our idea is to start from that project and evaluate some improvements such as:
Use NEAT for finding out the best topology for DQN
Try to 'give an help' to the agent putting into his replay experience some data derived from recordered plays made by humans (ourself)
Verify that performances can be improved by using Double Deep-Q learning
Assess the ability of the agent to play other levels of the same game
We've just beginned our work, let us know if our proposal is acceptable,


SONIC RL:
https://github.com/deepanshut041/Reinforcement-Learning/blob/master/cgames/05_sonic/sonic_dqn.ipynb

