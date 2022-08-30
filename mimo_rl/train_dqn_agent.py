'''
Illustrates reinforcement learning using the beam selection problem.
Use tensorboard to visualize training convergence.
Globecom Tutorial - December 7, 2021
Tutorial 29: Machine Learning for MIMO Systems with Large Arrays
Nuria Gonzalez-Prelcic (NCSU),
Aldebaro Klautau (UFPA) and
Robert W. Heath Jr. (NCSU)
'''
import numpy as np

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from env_mimo_rl_simple import Mimo_RL_Simple_Env

num_antenna_elements=32 #number of antennas at base station
grid_size=6 #world is a grid_size x grid_size grid
total_timesteps=80000 #training total steps

#defines the environment
env = Mimo_RL_Simple_Env(num_antenna_elements=num_antenna_elements, 
        grid_size=grid_size)

#use DQN
dqn_agent = DQN(policy="MlpPolicy",
                        batch_size=10, 
                        gamma=0.9,
                        verbose=1,
                        exploration_fraction=0.9,
                        learning_rate=0.01,
                        buffer_size=1500,
                        exploration_final_eps=0.02,
                        exploration_initial_eps=1.0,
                        #double_q = True,
                        #prioritized_replay = True,
                        learning_starts=100,
                        env=env,
                        tensorboard_log="./log_tensorboard/",
                        seed=0)

#train the agent
dqn_agent.learn(total_timesteps=total_timesteps)

#test it
#https://stable-baselines.readthedocs.io/en/master/common/evaluation.html
mean_reward, std_reward = evaluate_policy(dqn_agent, env, n_eval_episodes=20, return_episode_rewards=True)
print('evaluate_policy (mean and std)=')
print('mean=',mean_reward[0])
print('mean=',mean_reward[-1])
print('std=',std_reward)

#save agent to file
dqn_agent.save("beam_selection.dqn")
del dqn_agent

#load the trained agent and test it
trained_model = DQN.load("beam_selection.dqn")
env.enable_rendering() #allow visualizing
obs = env.reset() #reset environment
for i in range(10):
    action, _states = trained_model.predict(obs)
    obs, reward, dones, info = env.step(action)
    print('obs', obs)
    print('reward', reward)
    print('info', info)
