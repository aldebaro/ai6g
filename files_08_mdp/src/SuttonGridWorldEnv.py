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
'''
from __future__ import print_function
import numpy as np
import itertools
import gym
from gym import spaces
from random import choices, randint

from FiniteMDP import FiniteMDP
from NextStateProbabilitiesEnv import NextStateProbabilitiesEnv

'''
It is a subclass of NextStateProbabilitiesEnv. It is the
superclass that implements the step() function.
'''
class SuttonGridWorldEnv(NextStateProbabilitiesEnv):

    def __init__(self):
        self.__version__ = "0.1.0"
        #create the environment
        nextStateProbability, rewardsTable = self.create_environment()
        super().__init__(nextStateProbability, rewardsTable) #superclass constructor

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        pass

    def createActionsDataStructures(self):
            possibleActions = ['L', 'U', 'R', 'D']
            dictionaryGetIndex = dict()
            listGivenIndex = list()
            for uniqueIndex in range(len(possibleActions)):
                dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
                listGivenIndex.append(possibleActions[uniqueIndex])
            return dictionaryGetIndex, listGivenIndex

    def createStatesDataStructures(self,WORLD_SIZE):
        '''Defines the states. Overrides default method from superclass. WORLD_SIZE is the axis dimension, horizontal or vertical'''
        bufferStateList = list(itertools.product(np.arange(WORLD_SIZE), repeat=2))
        N = len(bufferStateList) #number of states
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

    def create_environment(self):
        '''Define the MDP process. Overrides default method from superclass.'''

        WORLD_SIZE = 5

        self.stateDictionaryGetIndex, self.stateListGivenIndex = self.createStatesDataStructures(WORLD_SIZE)
        #top-left corner is [0, 0]
        A_POS = [0, 1]
        A_PRIME_POS = [4, 1]
        B_POS = [0, 3]
        B_PRIME_POS = [2, 3]
        self.discount = 0.9

        #world = np.zeros((WORLD_SIZE, WORLD_SIZE))

        # left, up, right, down
        # actions = ['north', 'south', 'east', 'west'] #according to the book
        actions = ['L', 'U', 'R', 'D'] #use a single letter for simplicity
        self.actionDictionaryGetIndex, self.actionListGivenIndex = self.createActionsDataStructures()

        S = WORLD_SIZE * WORLD_SIZE
        A = len(actions)
        #self.A = A
        #self.S = S

        # this is the original code from github
        actionProb = []
        for i in range(0, WORLD_SIZE):
            actionProb.append([])
            for j in range(0, WORLD_SIZE):
                actionProb[i].append(dict({'L': 0.25, 'U': 0.25, 'R': 0.25, 'D': 0.25}))
        # this is the original code from github
        nextState = []
        actionReward = []
        for i in range(0, WORLD_SIZE):
            nextState.append([])
            actionReward.append([])
            for j in range(0, WORLD_SIZE):
                next = dict()
                reward = dict()
                if i == 0:
                    next['U'] = [i, j]
                    reward['U'] = -1.0
                else:
                    next['U'] = [i - 1, j]
                    reward['U'] = 0.0

                if i == WORLD_SIZE - 1:
                    next['D'] = [i, j]
                    reward['D'] = -1.0
                else:
                    next['D'] = [i + 1, j]
                    reward['D'] = 0.0

                if j == 0:
                    next['L'] = [i, j]
                    reward['L'] = -1.0
                else:
                    next['L'] = [i, j - 1]
                    reward['L'] = 0.0

                if j == WORLD_SIZE - 1:
                    next['R'] = [i, j]
                    reward['R'] = -1.0
                else:
                    next['R'] = [i, j + 1]
                    reward['R'] = 0.0

                if [i, j] == A_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = A_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

                if [i, j] == B_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = B_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

                nextState[i].append(next)
                actionReward[i].append(reward)

        #print('nextState = ', nextState)
        #print('actionReward = ', actionReward)
        # now convert to our general and smarter :) format:
        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                nextsdic = nextState[i][j]  # this is a dictionary
                rdic = actionReward[i][j]  # another dictionary
                # get state index
                s = self.stateDictionaryGetIndex[(i, j)]
                for a in range(A):
                    (nexti, nextj) = nextsdic[actions[a]]
                    nexts = self.stateDictionaryGetIndex[(nexti, nextj)]
                    # After the agent chooses a state, the MDP “dynamics” is such that p(s’/s,a) is 1 to only one state and zero to the others
                    nextStateProbability[s, a, nexts] = 1
                    r = rdic[actions[a]]
                    rewardsTable[s, a, nexts] = r
        return nextStateProbability, rewardsTable

    def prettyPrint(self):
        print(self.stateListGivenIndex)
        print(self.actionListGivenIndex)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentObservation = randint(0, self.S - 1)
        return self.currentObservation

    def get_state(self):
        return self.currentObservation

    def old_step(self, action):
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
        action_label = self.actionListGivenIndex[action]
        print("DELETE:",action_label)
        ob = self.reset()
        reward = 1
        gameOver = False
        history = None
        return ob, reward, gameOver, history

'''
Reproduce Figures 3.2 and 3.5 from [Sutton, 2018], Examples 3.5 and 3.8, respectively.
'''
def reproduce_figures():    
    env = SuttonGridWorldEnv()
    mdp = FiniteMDP(env)

    #how to get only the p(s'/s) by marginalizing p(s'/a,s) (summing dimension i=1)
    #onlyNextStateProbability = np.sum(mdp.nextStateProbability, 1)

    #get Fig. 3.5, which used a random policy
    equiprobable_policy = mdp.getEquiprobableRandomPolicy()
    state_values, iteration = mdp.compute_state_values(equiprobable_policy, in_place=True)
    print('Reproducing Fig. 3.2 from [Sutton, 2018] with equiprobable random policy in page 60.')
    print('Figure 3.2 Gridworld example: exceptional reward dynamics (left) and state-value function for the equiprobable random policy (right).')
    print('Number of iterations = ', iteration)
    print('State values:')
    print(np.round(np.reshape(state_values, (5,5)), 1))

    state_values, iteration = mdp.compute_optimal_state_values()
    print('Reproducing Fig. 3.5 from [Sutton, 2018] with optimum policy in page 65.')
    print('Figure 3.5: Optimal solutions to the gridworld example')
    print('Number of iterations = ', iteration)
    print('State values:')
    print(np.round(np.reshape(state_values, (5,5)), 1))

    #use the value-based policy to obtain the \pi_star right subplot in Fig. 3.5.
    action_values, iteration = mdp.compute_optimal_action_values()
    if False: #this is not shown in Fig. 3.5, but you can visualize action_values if you wish 
        print('iteration = ', iteration, ' action_values = ', np.round(action_values, 1))
    policy = mdp.convert_action_values_into_policy(action_values)
    mdp.prettyPrintValues(policy, env.stateListGivenIndex, env.actionListGivenIndex)

def try_q_learning():
    env = SuttonGridWorldEnv()
    mdp = FiniteMDP(env)
    num_iterations_for_training = 500000 #500000 leads to finding optimum policy
    num_iterations_for_testing = 10000
    #use optimum policy
    action_values, iteration = mdp.compute_optimal_action_values()
    optimum_policy = mdp.convert_action_values_into_policy(action_values)
    totalReward = mdp.run_MDP_for_given_policy(optimum_policy,maxNumIterations=num_iterations_for_testing)
    print('Using optimum policy, total reward=', totalReward)

    #learn a policy with Q-learning
    mdp = FiniteMDP(env)
    stateActionValues, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=1, maxNumIterationsQLearning=num_iterations_for_training, explorationProbEpsilon=0.2)
    print('stateActionValues:',stateActionValues)
    print('rewardsQLearning:',rewardsQLearning)
    #print('Using Q-learning, total reward over training=',np.sum(rewardsQLearning))
    qlearning_policy = mdp.convert_action_values_into_policy(stateActionValues)
    totalReward = mdp.run_MDP_for_given_policy(qlearning_policy,maxNumIterations=num_iterations_for_testing)
    print('Using Q-learning, total reward over test=', totalReward)
    print('Check the Q-learning policy:')
    mdp.prettyPrintValues(qlearning_policy, env.stateListGivenIndex, env.actionListGivenIndex)

if __name__ == '__main__':
    #env.prettyPrint()
    reproduce_figures() #From Sutton's book
    #try_q_learning()

