from config import SINGLE_ENV_PATH, DISCOUNT, TAU

from unity import *
from utils import ddpg

from ddpg_agent import Agent


env = Env(SINGLE_ENV_PATH)

brain_name = env.get_brain_names()
brain = env.get_brain()

env_info = env.reset_environment(train = False)

print("Current state : ", env.get_state())

input_shape = env.num_states()
output_shape = env.num_actions()


print(f"Num states : {env.num_states()}")
print(f"Num Actions : {env.num_actions()}")
print(f"Num Agents : {env.num_agents()}")

agent = Agent(input_shape, output_shape, TAU)

print("hello")

# reinforce(env, agent, n_episodes=1000, max_t=1000, gamma=DISCOUNT, print_every=4)
ddpg(agent, env, n_episodes=2000, max_t=700)

env.close()
