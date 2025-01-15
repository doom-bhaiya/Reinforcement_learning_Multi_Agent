import torch
from modelling import Actor_Critic

import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random
import numpy as np

from config import BUFFER_SIZE, BATCH_SIZE, LR, WEIGHT_DECAY, DISCOUNT

class Agent:

    def __init__(self, input_size, output_size, tau):

        self.actor_critic_local = Actor_Critic(input_size, output_size)
        self.actor_critic_target = Actor_Critic(input_size, output_size)

        self.actor_critic_optimizer = optim.Adam(self.actor_critic_local.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.TAU = tau

        self.memory = ReplayBuffer(output_size, BUFFER_SIZE, BATCH_SIZE)

    def step(self, state, action, reward, next_state, done):
        if reward:
            # print(reward)

            self.memory.add(state, action, reward, next_state, done)

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, DISCOUNT)

    def act(self, state, eps = 0.1):
        state = torch.from_numpy(state).float()
        self.actor_critic_local.eval()
        with torch.no_grad():
            action, _ = self.actor_critic_local(state) 
            

        if random.random() > eps:
            # self.actor_critic_local.train()
            # print(action)
            action = action.data.numpy() 
            # print(f"\r action = {action}") 
            return action
        else:
            tensor = torch.randn(1, 4)
            tensor = torch.tanh(tensor).data.numpy()
            return tensor  

    
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        _, Q_targets_next = self.actor_critic_target(next_states)
        Q_targets = rewards + (gamma * Q_targets_next)

        _, Q_expected = self.actor_critic_target(states)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # print(actions)
        # print(Q_expected)
        # print(critic_loss)

        self.actor_critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.actor_critic_optimizer.step()

        actor_loss = (-1 * Q_expected).mean()

        # self.actor_critic_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_critic_optimizer.step()

        self.soft_update(self.actor_critic_local, self.actor_critic_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)