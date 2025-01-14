from unityagents import UnityEnvironment
import numpy as np
import time


class Env:

    def __init__(self, filename):
        self.env = UnityEnvironment(file_name=filename)

    def get_brain_names(self):
        self.brain_names = self.env.brain_names[0]
        return self.brain_names

    def get_brain(self):
        self.brain = self.env.brains[self.brain_names]
        return self.brain
    
    def reset_environment(self, train = True):
        self.env_info = self.env.reset(train_mode=train)[self.brain_names]
        return self.env_info

    def num_agents(self):
        return len(self.env_info.agents)

    def num_actions(self):
        return self.brain.vector_action_space_size
    
    def get_state(self):
        self.current_state = self.env_info.vector_observations
        return self.current_state 
    
    def num_states(self):
        return self.current_state.shape[1]
    
    def step(self, action):

        # print(action[0])
        # action = action.astype("float64")

        # actions = np.random.randn(1, 4) # select an action (for each agent)
        # actions = np.array([[-1.0,-1.0,-1.0,-1.0]])
        # print(type(actions))
        # print(type(action))
        # print(actions.dtype)
        # print(action.dtype)
        # actions = np.clip(actions, -1, 1)

        # action = np.array([[ 1. ,        -0.36459392, -0.97171749,  1.        ]])
        self.env_info = self.env.step(action)[self.brain_names]   
        # time.sleep(0.1)
        next_states = self.env_info.vector_observations        
        rewards = self.env_info.rewards                        
        dones = self.env_info.local_done

        return next_states, rewards[0], dones

    def close(self):
        self.env.close()
