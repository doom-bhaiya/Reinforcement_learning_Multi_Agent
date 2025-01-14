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
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.train = train

    def act(self, state, eps=0.01):

        state = torch.from_numpy(state).float().to(self.device)
        state.requires_grad_(True)
        action_values = self.policy(state)
        self.policy.train()

        if random.random() > eps:
            return action_values
        else:
            action_values = ((torch.randn(1, 4) - 0.5) * 2).requires_grad_(True).to(self.device)
            return action_values


def reinforce(environment, agent, n_episodes=1000, max_t=1000, gamma=1.0, print_every=10):
    
    environment.reset_environment(train = False)
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        environment.reset_environment()
        state = environment.get_state()
        num_step = 0
        for t in range(max_t):
            log_prob = agent.act(state)
            saved_log_probs.append(log_prob)
            log_probs = log_prob.detach().numpy()
            state, reward, done = environment.step(log_probs)
            rewards.append(reward)
            rewards_exp = torch.tensor(reward).expand_as(log_prob)
            num_step += 1


            # policy_loss = torch.tensor(- log_prob * rewards_exp, requires_grad = True)
            # policy_loss = policy_loss.sum()
            
            # agent.optimizer.zero_grad()
            # policy_loss.backward()
            # agent.optimizer.step()
            if (num_step % 10 == 0):
                discounts = [gamma**i for i in range(len(rewards[-10 : ])+1)]
                R = sum([a*b for a,b in zip(discounts, rewards[-10 : ])])
                
                policy_loss = []
                for log_prob in saved_log_probs[-10 : ]: 
                    policy_loss.append(-log_prob * R)
                policy_loss = torch.cat(policy_loss).sum()
                
                agent.optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                agent.optimizer.step()

            if not done:
                break 
            
        scores_deque.append(np.sum(rewards))
        scores.append(sum(rewards))
        
        # discounts = [gamma**i for i in range(len(rewards)+1)]
        # print(rewards)

        # R = sum([a*b for a,b in zip(discounts, rewards)])
        # print(R)
        
        # policy_loss = []
        # for log_prob in saved_log_probs: 
        #     policy_loss.append(- log_prob * R)
        # policy_loss = torch.cat(policy_loss).sum()
        
        # agent.optimizer.zero_grad()
        # policy_loss.backward()
        # agent.optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores