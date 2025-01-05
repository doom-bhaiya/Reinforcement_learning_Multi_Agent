from unityagents import UnityEnvironment
import numpy as np

def get_env(path):
    return UnityEnvironment(file_name=path)

def get_brain_names(env):
    return env.brain_names[0]

def get_brain(env, brain_name):
    return env.brains[brain_name]

def reset_environment(env, brain_name, train = True):
    return env.reset(train_mode=True)[brain_name]

def num_agents(env_info):
    return len(env_info.agents)

def num_actions(brain):
    return brain.vector_action_space_size

def num_states(env_info):
    states = env_info.vector_observations
    return states.shape[1]

