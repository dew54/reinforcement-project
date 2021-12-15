import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from ..utils.replay_buffer import ReplayBuffer

class DQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model, loadModel = None, args = None):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        self.args = args
        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
       # self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0
        
        if loadModel == 'load_pt':
            self.state_dict = torch.load('traindParameters.pt')
            self.policy_net.load_state_dict(self.state_dict)
            self.policy_net.eval()
        elif loadModel == 'load_model':
            self.policy_net = torch.load('trainedCNN.model')
            self.policy_net.eval()
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                if self.args["useDDQN"]:
                #    print("Double DQN")
                    self.learnDouble(experiences)
                else:
                #    print("Vanilla DQN")
                    self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        current_Q = self.policy_net(states)
        current_Q = current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_Q = self.policy_net(next_states)

        max_next_Q = torch.max(next_Q, 1)[0]
        
        # Compute Q targets for current states 
        expected_Q  = rewards + (self.gamma * max_next_Q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_Q, expected_Q.detach())

        # Minimize the loss and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learnDouble(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)


    def soft_update(self, policy_model, target_model, tau):
        
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

    def saveNetwork(self):
        torch.save(self.target_net, 'trainedCNN.model')