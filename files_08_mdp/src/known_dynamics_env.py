'''
Gym env using two matrices: next state probabilities (nextStateProbability) and rewards table.
The initial state determined by reset() is random: any possible state.

This implementation is very general in the sense that it represents internally
rewards, actions and states as natural numbers. For instance, if a grid-world
has actions "left", "right", "up", "down", they must be mapped to integers such
as 0, 1, 2 and 3.

The finite MDP class is constructed based on an environment, which is an OpenAI's
gym.Env with spaces.Discrete() for both states (called observations in gym)
and actions. Only this environment can have knowledge about the labels associated
to the natural numbers used within this MDP class (in the grid-world example,
the labels "left", "right", "up", "down"). In this case, the environment provides
the label information via the lists: stateListGivenIndex and actionListGivenIndex.

The matrix nextStateProbability indicates p(s'/s,a). It can specify that actions are invalid in
a given state by having only zeros for a given pair (s,a). For instance, assuming S=2 states and
A=3 actions, the matrix
nextStateProbability= [[[0.1 0.9]
  [1.  0. ]
  [0.  0. ]]   <=== This indicates that action a=2 is invalid while in state 0.
 [[0.6 0.4]
  [0.  1. ]
  [1.  0. ]]]

Note that if these two matrices are assumed to be "correct" (representing the actual world),
then one can calculate the optimum solution using the methods in FiniteMDP.

@TODO:
 - support Moore (rewards associated to states) instead of only Mealy (rewards associated to transitions) (see https://www.youtube.com/watch?v=YiQxeuB56i0)
 - ??? use a single 5-dimension matrice instead of two matrices (see Example Exercise 3.4 "Give a table analogous to that in Example 3.3)
 - move code that generates trajectories to a class of agent, given that an environment do not create actions
 - support sparse matrices: e.g. assume the next state probability is sparse or not
 - use objects already available on gym.Env such as env.unwrapped.get_action_meanings()

Guidelines to create a gym env:
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
'''
import numpy as np
from random import choices, randint
import gymnasium as gym
from gymnasium import spaces


class KnownDynamicsEnv(gym.Env):
    def __init__(self, nextStateProbability, rewardsTable,
                 action_association='state', states_info=None, actions_info=None):
        super(KnownDynamicsEnv, self).__init__()
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable  # expected rewards

        if action_association == 'state':
            x = 1  # @TO-DO
        elif action_association == 'transition':
            x = 2  # @TO-DO
        else:
            raise Exception(
                "Invalid option. action_association must be state or transition.")

        self.current_observation_or_state = 0

        # (S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = nextStateProbability.shape[0]  # number of states
        self.A = nextStateProbability.shape[1]  # number of actions

        self.possible_states = np.arange(self.S)

        # initialize possible states and actions
        # we need to indicate only valid actions for each state
        # create a list of lists, that indicates for each state, the list of allowed actions
        self.possible_actions_per_state = self.get_valid_next_actions()

        # similar for states
        self.valid_next_states = self.get_valid_next_states()

        self.action_space = spaces.Discrete(self.A)
        # states are called observations in gym
        self.observation_space = spaces.Discrete(self.S)

        if actions_info == None:
            # initialize with default names and structures
            self.actionDictionaryGetIndex, self.actionListGivenIndex = createDefaultDataStructures(
                self.A, "A")
        else:
            self.actionDictionaryGetIndex = actions_info[0]
            self.actionListGivenIndex = actions_info[1]

        if states_info == None:
            # initialize with default names and structures
            self.stateDictionaryGetIndex, self.stateListGivenIndex = createDefaultDataStructures(
                self.S, "S")
        else:
            self.stateDictionaryGetIndex = states_info[0]
            self.stateListGivenIndex = states_info[1]

        self.currentIteration = 0
        self.reset()

    def get_valid_next_states(self) -> list:
        '''
        Pre-compute valid next states.
        '''
        # creates a list of lists
        valid_next_states = list()
        for s in range(self.S):
            valid_next_states.append(list())
            for a in range(self.A):
                for nexts in range(self.S):
                    p = self.nextStateProbability[s, a, nexts]
                    if p != 0:
                        # allow to have duplicated entries
                        valid_next_states[s].append(nexts)
        # eliminate duplicated entries
        for s in range(self.S):
            # convert to set
            valid_next_states[s] = set(valid_next_states[s])
            # convert back to list again
            valid_next_states[s] = list(valid_next_states[s])
        return valid_next_states

    def get_valid_next_actions(self) -> list:
        '''
        Pre-compute valid next actions.
        '''
        possible_actions_per_state = list()
        for s in range(self.S):
            possible_actions_per_state.append(list())
            for a in range(self.A):
                sum_for_s_a_pair = np.sum(self.nextStateProbability[s, a])
                if sum_for_s_a_pair > 0:
                    possible_actions_per_state[s].append(a)
        return possible_actions_per_state

    def step(self, action: int):
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

        # check if the chosen action is within the set of valid actions for that state
        valid_actions = self.possible_actions_per_state[s]
        if not (action in valid_actions):
            raise Exception("Action " + str(action) +
                            " is not in valid actions list: " + str(valid_actions))

        # find next state
        weights = self.nextStateProbability[s, action]
        nexts = choices(self.possible_states, weights, k=1)[0]

        # find reward value
        reward = self.rewardsTable[s, action, nexts]

        gameOver = False  # this is a continuing FMDP that never ends

        # history version with actions and states, not their indices
        history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
                   "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}

        # update for next iteration
        self.currentIteration += 1  # update counter
        self.current_observation_or_state = nexts

        # state is called observation in gym API
        ob = nexts
        return ob, reward, gameOver, history

    def postprocessing_MDP_step(env, history: dict, printPostProcessingInfo: bool):
        '''This method can be overriden by subclass and process history'''
        pass  # no need to do anything here

    def get_state(self) -> int:
        """Get the current observation."""
        return self.current_observation_or_state

    def reset(self) -> int:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentIteration = 0
        # note there are several versions of randint!
        self.current_observation_or_state = randint(0, self.S - 1)
        return self.current_observation_or_state

    # from gym.utils import seeding
    # def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]


class RandomKnownDynamicsEnv(KnownDynamicsEnv):
    '''
    Initialize a matrix with probability distributions.
    '''

    def __init__(self, S: int, A: int):
        nextStateProbability = self.init_random_next_state_probability(S, A)
        # these rewards can be negative
        rewardsTable = np.random.randn(S, A, S)
        self.__version__ = "0.1.0"
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def init_random_next_state_probability(self, S: int, A: int) -> np.ndarray:
        nextStateProbability = np.random.rand(S, A, S)  # positive numbers
        # for each pair of s and a, force numbers to be a probability distribution
        for s in range(S):
            for a in range(A):
                sum = np.sum(nextStateProbability[s, a])
                if sum == 0:
                    raise Exception(
                        "Sum is zero. np.random.rand did not work properly?")
                nextStateProbability[s, a] /= sum
        return nextStateProbability


class SimpleKnownDynamicsEnv(KnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        # make it a "left-right" Markov process without skips:
        # the state index cannot decrease over time nor skip
        # over the next state
        nextStateProbability = np.array([[[0.5, 0.5, 0],
                                          [0.9, 0.1, 0]],
                                         [[0, 0.5, 0.5],
                                          [0, 0.2, 0.8]],
                                         [[0, 0, 1],
                                          [0, 0, 1]]])
        rewardsTable = np.array([[[-3, 0, 0],
                                  [-2, 5, 5]],
                                 [[4, 5, 0],
                                  [2, 2, 6]],
                                 [[-8, 2, 80],
                                  [11, 0, 3]]])
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def reset(self) -> int:
        # make sure initial state is 0
        super().reset()
        self.current_observation_or_state = 0
        return self.current_observation_or_state


class RecycleRobotEnv(KnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        nextStateProbability, rewardsTable = self.recycle_robot_matrices()
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def recycle_robot_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        S = 2  # number of states
        A = 3  # number of actions
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
        nextStateProbability = np.zeros([S, A, S])
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

        rewardsTable = np.zeros([S, A, S])  # these rewards can be negative
        rewardsTable[high, search, high] = rsearch
        rewardsTable[high, search, low] = rsearch
        rewardsTable[low, search, high] = -3
        rewardsTable[low, search, low] = rsearch
        rewardsTable[high, wait, high] = rwait
        rewardsTable[low, wait, low] = rwait

        return nextStateProbability, rewardsTable


def createDefaultDataStructures(num_actions, prefix) -> tuple[dict, list]:
    possibleActions = list()
    for uniqueIndex in range(num_actions):
        possibleActions.append(prefix + str(uniqueIndex))
    dictionaryGetIndex = dict()
    listGivenIndex = list()
    for uniqueIndex in range(num_actions):
        dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
        listGivenIndex.append(possibleActions[uniqueIndex])
    return dictionaryGetIndex, listGivenIndex


if __name__ == "__main__":
    # Choose: 1) simple matrices, 2) recycle robot from Sutton's textbook, 3) random and 4) very simple
    chosen_example = 2
    # np.random.seed(110) # enable to repeat experiments
    if chosen_example == 1:
        env = SimpleKnownDynamicsEnv()
    elif chosen_example == 2:
        env = RecycleRobotEnv()
    elif chosen_example == 3:  # random initialization
        S = 3  # number of states
        A = 2  # number of actions
        env = RandomKnownDynamicsEnv(S, A)
    else:
        raise Exception("Not available")
    # about environment
    print("About environment:")
    print("Num of states =", env.S)
    print("Num of actions =", env.A)
    print("env.possible_actions_per_state", env.possible_actions_per_state)

    # one single action
    action = 0
    print("output of env.step function as defined by OpenAI Gym API (ob, reward, gameOver, history):", env.step(action))
    print("env current state:", env.get_state())
