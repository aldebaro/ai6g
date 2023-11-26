#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
'''
Implements the Grid World of Sutton & Barto's book, version 2018, with 550 pages:
Example 3.5: Gridworld, pag. 60 and Example 3.8: Solving the Gridworld, pag. 65.
Grid-world has 5 x 5 = 25 states.
The states are numbered from top-left (state 0) to bottom-right (state 24) in
zigzag scan:
0   1  2  3  4
5   6  7  8  9
...
20 21 23 23 24

This is a continuing (not episodic) process. It never ends.
Modified by Aldebaro. 2023
'''
from __future__ import print_function
import numpy as np
import itertools
import gymnasium
from gymnasium import spaces
from random import choices, randint

from known_dynamics_env import KnownDynamicsEnv
import finite_mdp_utils as fmdp


class SuttonGridWorldEnv(KnownDynamicsEnv):
    '''
    It is a subclass of NextStateProbabilitiesEnv. It is the
    superclass that implements the step() function.
    '''

    def __init__(self):
        self.__version__ = "0.1.0"
        # define grid size
        WORLD_SIZE = 5  # grid is WORLD_SIZE x WORLD_SIZE

        # create data structures (dic and list) to map names into indices for actions
        actionDictionaryGetIndex, actionListGivenIndex = self.createActionsDataStructures()
        actions_info = [actionDictionaryGetIndex, actionListGivenIndex]

        # create data structures (dic and list) to map names into indices for states
        stateDictionaryGetIndex, stateListGivenIndex = self.createStatesDataStructures(
            WORLD_SIZE)
        states_info = [stateDictionaryGetIndex, stateListGivenIndex]

        # create the environment
        nextStateProbability, rewardsTable = self.create_environment(
            WORLD_SIZE, stateDictionaryGetIndex)

        # superclass constructor
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable,
                                  actions_info=actions_info, states_info=states_info)

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        pass

    def createActionsDataStructures(self):
        '''
        Nice names for the actions.
        '''
        possibleActions = ['L', 'U', 'R', 'D']
        dictionaryGetIndex = dict()
        listGivenIndex = list()
        for uniqueIndex in range(len(possibleActions)):
            dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
            listGivenIndex.append(possibleActions[uniqueIndex])
        return dictionaryGetIndex, listGivenIndex

    def createStatesDataStructures(self, WORLD_SIZE):
        '''
        Nice names for states
        '''
        '''Defines the states. Overrides default method from superclass. WORLD_SIZE is the axis dimension, horizontal or vertical'''
        bufferStateList = list(itertools.product(
            np.arange(WORLD_SIZE), repeat=2))
        N = len(bufferStateList)  # number of states
        stateListGivenIndex = list()
        stateDictionaryGetIndex = dict()
        uniqueIndex = 0
        # add states to both dictionary and its inverse mapping (a list)
        for i in range(N):
            stateListGivenIndex.append(bufferStateList[i])
            stateDictionaryGetIndex[bufferStateList[i]] = uniqueIndex
            uniqueIndex += 1
        if False:
            print('stateDictionaryGetIndex = ', stateDictionaryGetIndex)
            print('stateListGivenIndex = ', stateListGivenIndex)
        return stateDictionaryGetIndex, stateListGivenIndex

    def create_environment(self, WORLD_SIZE, stateDictionaryGetIndex):
        '''Define the MDP process. Overrides default method from superclass.'''

        # top-left corner is [0, 0]
        A_POS = [0, 1]
        A_PRIME_POS = [4, 1]
        B_POS = [0, 3]
        B_PRIME_POS = [2, 3]

        # left, up, right, down
        # actions = ['north', 'south', 'east', 'west'] #according to the book
        # use a single letter for simplicity
        list_of_actions = ['L', 'U', 'R', 'D']

        S = WORLD_SIZE * WORLD_SIZE
        A = len(list_of_actions)

        # this is the representation adopted in the original code by Zhang et al,
        # from github, to describe moving in the grid.
        # They have adopted 2 lists of dictionaries:
        nextState_for_each_row = []
        actionReward_for_each_row = []
        for i in range(WORLD_SIZE):
            nextState_for_each_row.append([])
            actionReward_for_each_row.append([])
            for j in range(WORLD_SIZE):
                next_move_for_this_row = dict()
                reward_for_this_row = dict()
                if i == 0:  # top row
                    next_move_for_this_row['U'] = [
                        i, j]  # stay at the same place
                    reward_for_this_row['U'] = -1.0
                else:
                    next_move_for_this_row['U'] = [
                        i - 1, j]  # go up if not top row
                    reward_for_this_row['U'] = 0.0

                if i == WORLD_SIZE - 1:  # bottom (last) row
                    next_move_for_this_row['D'] = [
                        i, j]  # stay at the same place
                    reward_for_this_row['D'] = -1.0
                else:
                    next_move_for_this_row['D'] = [
                        i + 1, j]  # go down if not bottow row
                    reward_for_this_row['D'] = 0.0

                if j == 0:  # most-left column
                    next_move_for_this_row['L'] = [i, j]
                    reward_for_this_row['L'] = -1.0
                else:
                    next_move_for_this_row['L'] = [i, j - 1]
                    reward_for_this_row['L'] = 0.0

                if j == WORLD_SIZE - 1:  # most-right column
                    next_move_for_this_row['R'] = [i, j]
                    reward_for_this_row['R'] = -1.0
                else:
                    next_move_for_this_row['R'] = [i, j + 1]
                    reward_for_this_row['R'] = 0.0

                # we now update things to take in account the special positions A and B
                if [i, j] == A_POS:
                    # go from A_POS to A_PRIME_POS independent of the action
                    # (like a wormhole in universe: https://en.wikipedia.org/wiki/Wormhole)
                    next_move_for_this_row['L'] = next_move_for_this_row[
                        'R'] = next_move_for_this_row['D'] = next_move_for_this_row['U'] = A_PRIME_POS
                    reward_for_this_row['L'] = reward_for_this_row['R'] = reward_for_this_row['D'] = reward_for_this_row['U'] = 10.0

                if [i, j] == B_POS:
                    # go from B_POS to B_PRIME_POS independent of the action
                    next_move_for_this_row['L'] = next_move_for_this_row[
                        'R'] = next_move_for_this_row['D'] = next_move_for_this_row['U'] = B_PRIME_POS
                    reward_for_this_row['L'] = reward_for_this_row['R'] = reward_for_this_row['D'] = reward_for_this_row['U'] = 5.0

                nextState_for_each_row[i].append(next_move_for_this_row)
                actionReward_for_each_row[i].append(reward_for_this_row)

        # now convert the original representation to our general format:
        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                nextsdic = nextState_for_each_row[i][j]  # this is a dictionary
                rdic = actionReward_for_each_row[i][j]  # another dictionary
                # get state index
                s = stateDictionaryGetIndex[(i, j)]
                for a in range(A):
                    (nexti, nextj) = nextsdic[list_of_actions[a]]
                    nexts = stateDictionaryGetIndex[(nexti, nextj)]
                    # After the agent chooses a state, the MDP “dynamics” is such that p(s’/s,a) is 1 to only one state and zero to the others
                    # all other values are zero
                    nextStateProbability[s, a, nexts] = 1
                    r = rdic[list_of_actions[a]]
                    rewardsTable[s, a, nexts] = r
        return nextStateProbability, rewardsTable


def reproduce_figures():
    '''
    Reproduce Figures 3.2 and 3.5 from [Sutton, 2018], Examples 3.5 and 3.8, respectively.
    '''
    env = SuttonGridWorldEnv()

    # get Fig. 3.5, which used a uniform policy
    equiprobable_policy = fmdp.get_uniform_policy_for_fully_connected(
        env.S, env.A)
    state_values, iteration = fmdp.compute_state_values(
        env, equiprobable_policy, in_place=True)
    print(
        'Reproducing Fig. 3.2 from [Sutton, 2018] with equiprobable random policy in page 60.')
    print('Figure 3.2 Gridworld example: exceptional reward dynamics (left) and state-value function for the equiprobable random policy (right).')
    print('Number of iterations = ', iteration)
    print('State values:')
    print(np.round(np.reshape(state_values, (5, 5)), 1))

    state_values, iteration = fmdp.compute_optimal_state_values(env)
    print(
        'Reproducing Fig. 3.5 from [Sutton, 2018] with optimum policy in page 65.')
    print('Figure 3.5: Optimal solutions to the gridworld example')
    print('Number of iterations = ', iteration)
    print('State values:')
    print(np.round(np.reshape(state_values, (5, 5)), 1))

    # use the value-based policy to obtain the \pi_star right subplot in Fig. 3.5.
    tolerance = 0  # execute until full convergence
    action_values, stopping_criteria = fmdp.compute_optimal_action_values(
        env, tolerance=tolerance)
    print('Stopping criteria until convergence =', stopping_criteria)

    if False:  # this is not shown in Fig. 3.5, but you can visualize action_values if you wish
        print('iteration = ', iteration, ' stopping_criterion=', stopping_criterion, ' action_values = ',
              np.round(action_values, 1))
    policy = fmdp.convert_action_values_into_policy(action_values)
    fmdp.pretty_print_policy(env, policy)


if __name__ == '__main__':
    # env.prettyPrint()
    reproduce_figures()  # From Sutton's book

    env = SuttonGridWorldEnv()
    # This is a continuing (not episodic) process. It never ends.
    # we needed around 500000 Q-learning updates for finding optimum policy
    total_number_of_updates = 500000
    num_episodes = 500
    max_num_time_steps_per_episode = total_number_of_updates // num_episodes
    fmdp.compare_q_learning_with_optimum_policy(env,
                                                max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                                num_episodes=num_episodes,
                                                explorationProbEpsilon=0.2)
