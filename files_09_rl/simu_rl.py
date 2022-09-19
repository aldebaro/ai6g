import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.rl_simple import RLSimple
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.simple import SimpleTraffic

seed = 10
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
comm_env = CommunicationEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    "simple",
    rng=rng,
    obs_space=RLSimple.get_obs_space,
    action_space=RLSimple.get_action_space,
)
rl_agent = RLSimple(comm_env, 2, 2, np.array([2, 2]), seed=seed)
comm_env.set_agent_functions(
    rl_agent.obs_space_format, rl_agent.action_format, rl_agent.calculate_reward
)
check_env(comm_env)
total_number_steps = 10000
rl_agent.train(total_number_steps)

obs = comm_env.reset()
for step_number in tqdm(np.arange(total_number_steps)):
    sched_decision = rl_agent.step(obs)
    obs, _, end_ep, _ = comm_env.step(sched_decision)
    if end_ep:
        comm_env.reset()
