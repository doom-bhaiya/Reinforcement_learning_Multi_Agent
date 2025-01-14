from modelling import FullyConnectedModel
from collections import deque

import torch
import torch.optim as optim

import numpy as np
import random


class Agent:

    def __init__(self, input_size, output_size, train = False):

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = FullyConnectedModel(self.input_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.train = train

    def act(self, state, eps=0.2):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """ 
        # state = torch.from_numpy(state).float().to(self.device)
        # print(state)
        # print(torch.from_numpy(state).float().unsqueeze(0).shape)
        state = torch.from_numpy(state).float().to(self.device)
        state.requires_grad_(True)
        self.policy.eval()
        # with torch.no_grad():
        action_values = self.policy(state)
        self.policy.train()

        if random.random() > eps:
            return action_values
        else:
            action_values = ((torch.randn(1, 4) - 0.5) * 2).requires_grad_(True).to(self.device)
            # print(action_values)
            return action_values

        # if self.train == False:
        #     time.sleep(0.025)
        # Epsilon-greedy action selection
        # print(action_values)
        # return action_values


def reinforce(environment, agent, n_episodes=1000, max_t=1000, gamma=1.0, print_every=10):
    environment.reset_environment(train = False)
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        environment.reset_environment()
        state = environment.get_state()
        for t in range(max_t):
            log_prob = agent.act(state)
            saved_log_probs.append(log_prob)
            log_prob = log_prob.detach().numpy()
            # log_prob = log_prob.astype("float64")
            state, reward, done = environment.step(log_prob)
            rewards.append(reward)
            if not done:
                print(t)
                break 
            # print(t)
            
        scores_deque.append(np.sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]

        R = sum([a*b for a,b in zip(discounts, rewards)])
        print(R)
        
        policy_loss = []
        for log_prob in saved_log_probs: 
            policy_loss.append(- log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        agent.optimizer.zero_grad()
        policy_loss.backward()
        agent.optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores