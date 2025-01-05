from config import SINGLE_ENV_PATH

from unity import *

env = get_env(SINGLE_ENV_PATH)

brain_name = get_brain_names(env)
brain = get_brain(env, brain_name)

env_info = reset_environment(env, brain_name, train = False)

print(f"Num states : {num_states(env_info)}")
print(f"Num Actions : {num_actions(brain)}")
print(f"Num Agents : {num_agents(env_info)}")


env.close()
