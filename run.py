
import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import os
import sendEmail
# import google.cloud.logging

import socket
import threading

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import logging


# %matplotlib inline

import sys
# sys.path.append('../../')
from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame


# client = google.cloud.logging.Client()

logging.warning("Started")


def tcp_health_check():
    HOST = ''  # Empty string binds to all available interfaces
    PORT = 8080  # Choose a port number for your TCP health check

    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f'TCP health check server listening on port {PORT}')

        while True:
            conn, addr = s.accept()
            handle_health_check(conn)

def handle_health_check(conn):
    # Respond with a success status code
    conn.sendall(b'HTTP/1.1 200 OK\r\n\r\ntrue')
    conn.close()

# Start the TCP health check server in a separate thread
health_check_thread = threading.Thread(target=tcp_health_check)
health_check_thread.start()
logging.warning("Started Server")




logging.warning("Setting Up")

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()
plt.figure()
# plt.imshow(env.reset())
plt.title('Original Frame')
# plt.show()


possible_actions = {
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


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
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

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

env.viewer = None
# watch an untrained agent
state = stack_frames(None, env.reset(), True) 
# for j in range(200):
#     env.render(close=False)
#     action = agent.act(state, eps=0.01)
#     next_state, reward, done, _ = env.step(possible_actions[action])
#     state = stack_frames(state, next_state, False)
#     if done:
#         env.reset()
#         break 
# env.render(close=True)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
variable_name = "NUM_EPISODES"
num_episodes = os.environ.get(variable_name)

logging.warning("Ended setting Up")


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)

        # Punish the agent for not moving forward
        prev_state = {}
        steps_stuck = 0
        timestamp = 0

        while timestamp < 10000:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1

            # Punish the agent for standing still for too long.
            if (prev_state == info):
                steps_stuck += 1
            else:
                steps_stuck = 0
            prev_state = info
    
            if (steps_stuck > 20):
                reward -= 1
            
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        
        # clear_output(True)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(scores)), scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
    
    return scores

logging.warning("Starting training")

scores = train(int(num_episodes))

clear_output(True)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
# plt.show()

logging.warning("Saving results")

plt.savefig('score.png',transparent=True)

plt.plot([epsilon_by_epsiode(i) for i in range(int(num_episodes))])

plt.savefig('epsilon.png',transparent=True)


# scores = train(1000)

print("Trained!")

agent.saveNetwork()


def send_email(subject, message, sender, recipients, smtp_server, smtp_port, username, password, attachments=None):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    

    # Add the message body
    msg.attach(MIMEText(message, 'plain'))

    # Attach any files
    if attachments:
        for attachment in attachments:
            part = MIMEBase('application', 'octet-stream')
            with open(attachment, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{attachment}"')
            msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print("Error: Unable to send email.")
        print(e)


subject = "Hello from Python!"
message = "This is the body of the email."
sender = "dew54@live.it"
recipients = ["dvdvdm96@gmail.com"]
smtp_server = "smtp.office365.com"
smtp_port = 587
username = "dew54@live.it"
password = "mangiare"
attachments = ["score.png", "epsilon.png", "trainedCNN.model", "trainedParameters.pt"]  # Replace with actual file paths


logging.warning("Sending email")

send_email(subject, message, sender, recipients, smtp_server, smtp_port, username, password, attachments)

# health_check_thread.join()
