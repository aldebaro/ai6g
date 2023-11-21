'''
Gym env using probability matrices.
If the nextStateProbability is the correct one,
then one can calculate the optimum solution using the methods in FiniteMDP.

Guidelines to create a gym env:
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
'''
from xmlrpc.client import boolean
import numpy as np
from random import choices, randint, seed
#import gym
import gymnasium as gym
from gymnasium import spaces

class NextStateProbabilitiesEnv(gym.Env):
    def __init__(self, nextStateProbability, rewardsTable, action_association='state'):
        super(NextStateProbabilitiesEnv, self).__init__()
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable #expected rewards

        self.current_observation_or_state = 0

        #(S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = nextStateProbability.shape[0]  # number of states
        self.A = nextStateProbability.shape[1]  # number of actions

        self.possible_states = np.arange(self.S)

        # initialize possible states and actions
        self.possible_actions_per_state = [None]*(self.S)
        for s in range(self.S):
            self.possible_actions_per_state[0] = list()
            for a in range(self.A):
                sum_for_s_a_pair = np.sum(self.nextStateProbability[s,a])
                if sum_for_s_a_pair > 0:
                    self.possible_actions_per_state.append(a)

        print(self.possible_actions_per_state)
        exit(1)

        self.action_space = spaces.Discrete(self.A)
        self.observation_space = spaces.Discrete(self.S) #states are called observations in gym

        self.currentIteration = 0
        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : array of topN integers
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        s = self.get_state()

        # find next state
        weights = self.nextStateProbability[s, action]
        nexts = choices(self.possible_states, weights, k=1)[0]

        # find reward value
        reward = self.rewardsTable[s, action, nexts]

        gameOver = False
        if self.currentIteration > np.Inf:
            ob = self.reset()
            gameOver = True  # game ends
        else:
            ob = self.get_state()

        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}
        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "observation_tp1": self.stateListGivenIndex[self.get_state()]}
        self.currentIteration += 1 # update counter
        self.current_observation_or_state = nexts # state is called observation in gym API
        return ob, reward, gameOver, history

    def get_state(self):
        """Get the current observation."""
        return self.current_observation_or_state

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentIteration = 0
        # note there are several versions of randint!
        self.current_observation_or_state = randint(0, self.S - 1)
        return self.get_state()

    # from gym.utils import seeding
    # def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def generate_trajectory(self, num_steps):
        #if one assumes a uniform distribution for actions, as below, it may choose invalid actions
        possible_actions = np.asarray(self.possible_actions_per_state[self.current_observation_or_state])
        taken_actions = choices(possible_actions, k=num_steps)
        #instead, let us choose only valid actions


        taken_actions = np.array(taken_actions) # convert from list to numpy array
        rewards_tp1 = np.zeros(num_steps)
        states = np.zeros(num_steps)
        #rewards_tp1[0] = np.NaN # indicates that at time t=0 one obtains the reward for t=1 (convention of Sutton's book)
        for t in range(num_steps):
            states[t] = self.current_observation_or_state
            ob, reward, gameOver, history = self.step(taken_actions[t])
            rewards_tp1[t] = reward #at time t=0 one obtains the reward for t=1 (convention of Sutton's book)
        return taken_actions, rewards_tp1, states


    '''Estimate using Monte Carlo.
    '''
    def estimate_model_probabilities(env: gym.Env):
        pass


'''
Initialize a matrix with probability distributions.
'''
def init_random_next_state_probability(S, A):
    nextStateProbability = np.random.rand(S,A,S) #positive numbers
    #for each pair of s and a, force numbers to be a probability distribution
    for s in range(S):
        for a in range(A):
            sum = np.sum(nextStateProbability[s,a])
            if sum == 0:
                raise Exception("Sum is zero. np.random.rand did not work properly?")
            nextStateProbability[s,a] /= sum
    return nextStateProbability

'''
Format as a single vector
'''
def format_trajectory_as_single_array(taken_actions, rewards_tp1, states):
    T = len(taken_actions)
    trajectory = np.zeros(3*T) # pre-allocate space for S0, A0, R1, S1, A1, R2, S2, A2, R3, ...
    for t in range(T):
        trajectory[3*t] = states[t]
        trajectory[3*t+1] = taken_actions[t]
        trajectory[3*t+2] = rewards_tp1[t]
    return trajectory

def print_trajectory(trajectory):
    '''
    Indices allow to interpret the trajectory according to convention in Sutton & Barto's book'''
    T = len(trajectory) // 3
    t = 0
    print("time=" + str(t) + "  S" + str(t) + "=" + str(trajectory[3*t]) + "  A" + str(t) + "=" + str(trajectory[3*t+1]) + "  R in undefined")
    for t in range(1, T):
        print("time=" + str(t) + "  S" + str(t) + "=" + str(trajectory[3*t]) + "  A" + str(t) + "=" + str(trajectory[3*t+1]) + "  R" + str(t) + "=" + str(trajectory[3*(t-1)+2]))

def recycle_robot_matrices():
    S = 2 # number of states
    A = 3 # number of actions
    alpha = 0.1
    beta = 0.4
    rsearch = -1
    rwait = -2
    # states
    high = 0
    low = 1
    # actions
    search = 0
    wait = 1
    recharge = 2

    # nextStateProbability Table
    nextStateProbability = np.zeros([S,A,S])
    nextStateProbability[high, search, high] = alpha
    nextStateProbability[high, search, low] = 1-alpha
    nextStateProbability[low, search, high] = 1 - beta
    nextStateProbability[low, search, low] = beta
    nextStateProbability[high, wait, high] = 1
    nextStateProbability[high, wait, low] = 0
    nextStateProbability[low, wait, high] = 0
    nextStateProbability[low, wait, low] = 1
    nextStateProbability[low, recharge, high] = 1
    nextStateProbability[low, recharge, low] = 0

    rewardsTable = np.zeros([S,A,S]) #these rewards can be negative
    rewardsTable[high, search, high] = rsearch
    rewardsTable[high, search, low] = rsearch
    rewardsTable[low, search, high] = -3
    rewardsTable[low, search, low] = rsearch
    rewardsTable[high, wait, high] = rwait
    rewardsTable[low, wait, low] = rwait

    return nextStateProbability, rewardsTable

if __name__ == "__main__":
    S = 3 # number of states
    A = 2 # number of actions
    #np.random.seed(110)
    if False:
        # To do
        nextStateProbability = np.array([[[0, 0.5, 0.5],
                                          [0, 0.5, 0.5]],
                                         [[0, 0.5, 0.5],
                                          [0, 0.5, 0.5]],
                                          [[0, 1, 0],
                                          [0, 1, 0]]])
        rewardsTable = np.array([[[-3, 0, 0],
                                  [-2, 5, 5]],
                                  [[4, 5, 0],
                                   [2, 2, 6]],
                                   [[8, 3, 1],
                                    [11, 0, 3]]])
    elif True:
        nextStateProbability, rewardsTable = recycle_robot_matrices()
    else: # random initialization
        nextStateProbability = init_random_next_state_probability(S, A)
        rewardsTable = np.random.randn(S,A,S) #these rewards can be negative
    print("nextStateProbability=", nextStateProbability)
    print("rewardsTable=", rewardsTable)
    environment = NextStateProbabilitiesEnv(nextStateProbability, rewardsTable)
    # one single action
    action = 0
    print("output of env.step function:", environment.step(action))
    print("env current state:", environment.get_state())
    # several actions
    num_steps = 10
    taken_actions, rewards_tp1, states = environment.generate_trajectory(num_steps)
    trajectory = format_trajectory_as_single_array(taken_actions, rewards_tp1, states)
    print_trajectory(trajectory)
