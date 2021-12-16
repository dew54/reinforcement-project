# reinforcement-project
RL academic project

Paper on:
https://docs.google.com/document/d/1pDiRW2g1yKKiK6LilJiCd5ayzWb1l7AyQhKL6smqo44/edit?usp=sharing


git clone https://github.com/dew54/reinforcement-project.git



Note: 
Retro-gym is not compatible with python 3.9 or higher. Install python 3.7, then for installing packages for a specific version of python:
py -3.7 -m pip install gym-retro

To import a rom:
py -3.7 -m retro.import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'


TODO
1) Train agent with DQN base 
2) Use NEAT to find out the best DQN topology
3) Insert in replay experience human played input keys
4) Transfer learning on other game levels


SONIC RL:
https://github.com/deepanshut041/Reinforcement-Learning/blob/master/cgames/05_sonic/sonic_dqn.ipynb

COLAB script: // Just copy paste and run:
!git clone --branch colab https://github.com/dew54/reinforcement-project.git
!pip install gym-retro
import retro
!python -m retro.import 'reinforcement-project/rom'
!python reinforcement-project/test.py --foo bar

To keep colab running, paste in javascript console:

function ClickConnect(){
console.log("Working"); 
document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
}
var clicker = setInterval(ClickConnect,60000);