# reinforcement-project
RL academic project

Paper on:
https://docs.google.com/document/d/1pDiRW2g1yKKiK6LilJiCd5ayzWb1l7AyQhKL6smqo44/edit?usp=sharing

Note: 
Retro-gym is not compatible with python 3.9 or higher. Install python 3.7, then for installing packages for a specific version of python:
py -3.7 -m pip install gym-retro

To import a rom:
py -3.7 -m retro.import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'

per il training della nn, una registrazione di gioco di uno di noi sarà usata per il training e quella dell'altro per la validation

TODO
1) write pressed keys in csv
2) train NN with that csv
3) import NN structure and weights to feed NEAT
4) transfer learning on new levels
5) explain maths




SONIC RL:
https://github.com/deepanshut041/Reinforcement-Learning/blob/master/cgames/05_sonic/sonic_dqn.ipynb



!git clone --branch colab https://github.com/dew54/reinforcement-project.git
!pip install gym-retro
!python reinforcement-project/Sonic_run.py --foo bar