'''
This class implements a finite Markov Decision Process (MDP) for tabular RL.
Look at the definition of a finite MDP in page 49 of
Sutton & Barto's book, version 2018, with 550 pages.

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

Note that a policy is represented here as a matrix S x A, providing a distribution
over the possible actions for each state. A matrix with the state values can be
easily converted into a policy. 

Aldebaro. June 25, 2022.
@TODO should not assume it is a NextStateEnv, but support a general env. In
compute_optimal_action_values and others need to pass as parameters.
'''
from __future__ import print_function
import numpy as np
from builtins import print
# from scipy.stats import rv_discrete
from random import choices
#from akpy.NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
from src.NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
import gym
from gym import spaces

class FiniteMDP:
    #def __init__(self, environment: gym.Env, sparse = False):
    def __init__(self, environment: gym.Env):
        self.__version__ = "0.1.1"
        # print("AK Finite MDP - Version {}".format(self.__version__))

        #checks if env is a gym with discrete spaces
        assert isinstance(environment.action_space, spaces.Discrete)
        assert isinstance(environment.observation_space, spaces.Discrete)        

        #self.sparse = sparse #assume the next state probability is sparse or not
        self.environment = environment        
        self.S = environment.observation_space.n
        self.A = environment.action_space.n

        self.currentObservation = 0
        self.currentIteration = 0
        self.environment.reset()

    def prettyPrintValues(self, action_values, stateListGivenIndex, actionListGivenIndex):
        '''
        Note that a policy is represented here as a distribution over the
        possible actions for each state and stored as an array of dimension S x A.
        A general policy may be represented in other ways. But considering the
        adopted representation here, the "policy" coincides with the distribution
        of actions for each state.
        '''
        for s in range(self.S):
            currentState = stateListGivenIndex[s]
            print('\ns=', s, '=', currentState)  # ' ',end='')
            for a in range(self.A):
                if action_values[s, a] == 0:
                    continue
                currentAction = actionListGivenIndex[a]
                print(' | a=', a, '=', currentAction, end='')

    def getEquiprobableRandomPolicy(self):
        policy = np.zeros((self.S, self.A))
        uniformProbability = 1.0 / self.A
        for s in range(self.S):
            for a in range(self.A):
                policy[s, a] = uniformProbability
        return policy

    '''
    Iterative policy evaluation. Page 75 of [Sutton, 2018].
    Here a policy (not necessarily optimum) is provided.
    It can generate, for instance, Fig. 3.2 in [Sutton, 2018]
    '''
    def compute_state_values(self, policy, in_place=False, discountGamma = 0.9):
        S = self.S
        A = self.A
        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = new_state_values.copy()
        iteration = 1
        while True:
            src = new_state_values if in_place else state_values
            for s in range(S):
                value = 0
                for a in range(A):
                    if policy[s, a] == 0:
                        continue  # save computation
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        r = self.environment.rewardsTable[s, a, nexts]
                        value += policy[s, a] * p * (r + discountGamma * src[nexts])
                        # value += p*(r+discount*src[nexts])
                new_state_values[s] = value
            # AK-TODO, Sutton, end of pag. 75 uses the max of individual entries, while
            # here we are using the summation:
            improvement = np.sum(np.abs(new_state_values - state_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)
            if improvement < 1e-4:
                state_values = new_state_values.copy()
                break

            state_values = new_state_values.copy()
            iteration += 1

        return state_values, iteration

    '''
    Different than compute_state_values(), in this method a policy is not provided,
    and the optimum values are estimated. In [Sutton, 2018] the main result of
    this method in called "the optimal state-value function" and defined in
    Eq. (3.15) in page 62.

    For unknown reason, this implementation makes the
    RAM consumption increase indefinitely. Besides, it is
    slow because tries all states
    '''
    def buggy_compute_optimal_state_values(self, discountGamma = 0.9):
        '''Page 63 of [Sutton, 2018], Eq. (3.19)'''
        S = self.S
        A = self.A
        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = new_state_values.copy()
        iteration = 1
        while True:
            for s in range(S):
                print('AK, s=',s)
                a_candidates = list()
                for a in range(A):
                    value = 0
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        if p != 0:
                            r = self.environment.rewardsTable[s, a, nexts]
                            value += p * (r + discountGamma * state_values[nexts])
                    a_candidates.append(value)
                new_state_values[s] = np.max(a_candidates)
            improvement = np.sum(np.abs(new_state_values - state_values))
            print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)

            state_values = new_state_values.copy()
            if improvement < 1e-4:
                break

            iteration += 1

        return state_values, iteration

    def compute_optimal_state_values(self, discountGamma = 0.9):
        '''Page 63 of [Sutton, 2018], Eq. (3.19)'''
        S = self.S
        A = self.A
        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = np.zeros((S,))
        iteration = 1
        a_candidates = np.zeros((A,))
        while True:
            for s in range(S):
                #fill with zeros without creating new array
                #a_candidates[:] = 0
                a_candidates.fill(0.0)
                for a in range(A):
                    value = 0
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        if p != 0:
                            r = self.environment.rewardsTable[s, a, nexts]
                            value += p * (r + discountGamma * state_values[nexts])
                    a_candidates[a] = value
                new_state_values[s] = np.max(a_candidates)
            improvement = np.sum(np.abs(new_state_values - state_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)
            #I am avoiding to use np.copy() here because memory kept growing
            for i in range(S):
                state_values[i] = new_state_values[i]
            if improvement < 1e-4:
                break

            iteration += 1

        return state_values, iteration

    '''
    This is useful when nextStateProbability is sparse. It only goes over the
    next states that are feasible.
    '''
    def compute_optimal_state_values_sparse(self, valid_next_states, discountGamma = 0.9):
    #def compute_optimal_state_values_sparse_prob_matrix(self, discountGamma = 0.9):
        '''Page 63 of [Sutton, 2018], Eq. (3.19)'''
        S = self.S
        A = self.A

        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = np.zeros((S,))
        iteration = 1
        a_candidates = np.zeros((A,))
        while True:
            for s in range(S):
                #print(s)
                #fill with zeros without creating new array
                #a_candidates[:] = 0
                a_candidates.fill(0.0)
                feasible_next_states = valid_next_states[s]
                num_of_feasible_next_states = len(feasible_next_states)                
                for a in range(A):
                    value = 0
                    #for nexts in range(S):
                    for feasible_nexts in range(num_of_feasible_next_states):
                        #print('feasible_nexts=',feasible_nexts)
                        nexts = feasible_next_states[feasible_nexts]                        
                        p = self.environment.nextStateProbability[s, a, nexts]
                        r = self.environment.rewardsTable[s, a, nexts]
                        #print('p',p,'state_values[nexts]',state_values[nexts],'r',r)
                        value += p * (r + discountGamma * state_values[nexts])
                    a_candidates[a] = value
                new_state_values[s] = np.max(a_candidates)
            improvement = np.sum(np.abs(new_state_values - state_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)
            #I am avoiding to use np.copy() here because memory kept growing
            for i in range(S):
                state_values[i] = new_state_values[i]
            if improvement < 1e-4:
                break

            iteration += 1

        return state_values, iteration

    '''
    If next state probability is sparse, pre-compute the valid next states.
    '''
    def get_valid_next_states(self):
        S = self.S
        A = self.A
        #creates a list of lists
        valid_next_states = list()
        for s in range(S):
            valid_next_states.append(list())
            for a in range(A):
                for nexts in range(S):
                    p = self.environment.nextStateProbability[s, a, nexts]
                    if p != 0:
                        #allow to have duplicated entries
                        valid_next_states[s].append(nexts)
        #eliminate duplicated entries
        for s in range(S):
            #convert to set
            valid_next_states[s] = set(valid_next_states[s])
            #convert back to list again
            valid_next_states[s] = list(valid_next_states[s])
        return valid_next_states


    '''
    In [Sutton, 2018] the main result of this method in called "the optimal
    action-value function" and defined in Eq. (3.16) in page 63.
    '''
    def compute_optimal_action_values(self, discountGamma = 0.9):
        '''Page 64 of [Sutton, 2018], Eq. (3.20)'''
        S = self.S
        A = self.A
        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_action_values = np.zeros((S, A))
        action_values = new_action_values.copy()
        iteration = 1
        while True:
            # src = new_action_values if in_place else action_values
            for s in range(S):
                for a in range(A):
                    value = 0
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts] #p(s'|s,a) = p(nexts|s,a)
                        r = self.environment.rewardsTable[s, a, nexts] #r(s,a,s') = r(s,a,nexts)
                        best_a = -np.Infinity
                        for nexta in range(A):
                            temp = action_values[nexts, nexta]
                            if temp > best_a:
                                best_a = temp
                        value += p * (r + discountGamma * best_a)
                        # value += p*(r+discount*src[nexts])
                        # print('aa', value)
                    new_action_values[s, a] = value
            improvement = np.sum(np.abs(new_action_values - action_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', action_values)
                print('new state values=', new_action_values)
                print('it=', iteration, 'improvement = ', improvement)
            if improvement < 1e-4:
                action_values = new_action_values.copy()
                break

            action_values = new_action_values.copy()
            iteration += 1

        return action_values, iteration

    '''
    In [Sutton, 2018] the main result of this method in called "the optimal
    action-value function" and defined in Eq. (3.16) in page 63.

    Assumes sparsity.
    '''
    def compute_optimal_action_values_sparse(self, valid_next_states, discountGamma = 0.9):
        '''Page 64 of [Sutton, 2018], Eq. (3.20)'''
        S = self.S
        A = self.A
        #(S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_action_values = np.zeros((S, A))
        action_values = np.zeros((S, A))
        iteration = 1
        while True:
            # src = new_action_values if in_place else action_values
            for s in range(S):
                feasible_next_states = valid_next_states[s]
                num_of_feasible_next_states = len(feasible_next_states)                
                for a in range(A):
                    value = 0
                    #for nexts in range(S):
                    for feasible_nexts in range(num_of_feasible_next_states):
                        nexts = feasible_next_states[feasible_nexts]
                        p = self.environment.nextStateProbability[s, a, nexts] #p(s'|s,a) = p(nexts|s,a)
                        r = self.environment.rewardsTable[s, a, nexts] #r(s,a,s') = r(s,a,nexts)
                        best_a = -np.Infinity
                        for nexta in range(A):
                            temp = action_values[nexts, nexta]
                            if temp > best_a:
                                best_a = temp
                        value += p * (r + discountGamma * best_a)
                        # value += p*(r+discount*src[nexts])
                        # print('aa', value)
                    new_action_values[s, a] = value
            improvement = np.sum(np.abs(new_action_values - action_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', action_values)
                print('new state values=', new_action_values)
                print('it=', iteration, 'improvement = ', improvement)
            if improvement < 1e-4:
                #action_values = new_action_values.copy()
                #I am avoiding to use np.copy() here because memory kept growing
                for i in range(S):
                    for j in range(A):
                        action_values[i,j] = new_action_values[i,j]
                break

            #action_values = new_action_values.copy()
            #I am avoiding to use np.copy() here because memory kept growing
            for i in range(S):
                for j in range(A):
                    action_values[i,j] = new_action_values[i,j]

            iteration += 1

        return action_values, iteration

    def convert_action_values_into_policy(self, action_values):
        (S, A) = action_values.shape
        policy = np.zeros((S, A))
        for s in range(S):
            maxPerState = max(action_values[s])
            maxIndices = np.where(action_values[s] == maxPerState)
            # maxIndices is a tuple and we want to get first element maxIndices[0]
            policy[s, maxIndices] = 1.0 / len(maxIndices[0])  # impose uniform distribution
        return policy

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method can be overriden by subclass and process history'''
        pass  # no need to do anything here

    def run_MDP_for_given_policy(self, policy, maxNumIterations=100, printInfo=False, printPostProcessingInfo=False):
        self.environment.reset()
        s = self.environment.get_state()
        totalReward = 0
        #if printInfo:
            #print('Initial state = ', self.stateListGivenIndex[s])
        for it in range(maxNumIterations):
            myweights = np.squeeze(policy[s])
            sumWeights = np.sum(myweights) #AK-TODO: what if there are positive and negative numbers canceling out?
            if sumWeights == 0:
                myweights = np.ones(myweights.shape)
            if True:
                #AK-TODO the choices is giving troubles with negative weights. Make them all positive
                #AK-TODO this changes the relative importances / weights, right?
                minWeight = np.min(myweights)
                if minWeight < 0:
                    myweights += (-minWeight)+1e-30
                sumWeights = np.sum(myweights)
                myweights /= sumWeights
            action = choices(np.arange(self.A), weights=myweights, k=1)[0]
            ob, reward, gameOver, history = self.environment.step(action)

            #AK
            if reward < -5:
                print(history)

            self.postprocessing_MDP_step(history, printPostProcessingInfo)
            if printInfo:
                print(history)
            totalReward += reward
            s = self.environment.get_state()
            if gameOver == True:
                break
        if printInfo:
            print('totalReward = ', totalReward)
        return totalReward

    # an episode with Q-Learning
    # @stateActionValues: values for state action pair, will be updated
    # @expected: if True, will use expected Sarsa algorithm
    # @stepSize: step size for updating
    # @return: total rewards within this episode
    def q_learning(self, stateActionValues, maxNumIterationsQLearning=100, stepSizeAlpha=0.1,
                   explorationProbEpsilon=0.01, discountGamma = 0.9):
        currentState = self.environment.get_state()
        rewards = 0.0
        for numIterations in range(maxNumIterationsQLearning):
            currentAction = self.chooseAction(currentState, stateActionValues, explorationProbEpsilon)

            ob, reward, gameOver, history = self.environment.step(currentAction)
            newState = self.environment.get_state()
            reward = self.environment.rewardsTable[currentState, currentAction, newState]
            #reward = self.environment.get_current_reward()
            rewards += reward
            # Q-Learning update
            stateActionValues[currentState, currentAction] += stepSizeAlpha * (
                    reward + discountGamma * np.max(stateActionValues[newState, :]) -
                    stateActionValues[currentState, currentAction])
            currentState = newState
        # normalize rewards to facilitate comparison
        return rewards / maxNumIterationsQLearning

    # choose an action based on epsilon greedy algorithm
    def chooseAction(self, state, stateActionValues, explorationProbEpsilon=0.01):
        #print(state)
        if np.random.binomial(1, explorationProbEpsilon) == 1:
            return np.random.choice(np.arange(self.A))
        else:
            values_ = stateActionValues[state, :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    def execute_q_learning(self, maxNumIterations=100, maxNumIterationsQLearning=10, num_runs=1,
                           stepSizeAlpha=0.1, explorationProbEpsilon=0.01):
        '''Use independent runs instead of a single run.
        maxNumIterationsQLearning is used to smooth numbers'''

        rewardsQLearning = np.zeros(maxNumIterations)
        for run in range(num_runs):
            stateActionValues = np.zeros((self.S, self.A))
            for i in range(maxNumIterations):
                # update stateActionValues in-place
                reward = self.q_learning(stateActionValues,
                                         maxNumIterationsQLearning=maxNumIterationsQLearning,
                                         stepSizeAlpha=stepSizeAlpha,
                                         explorationProbEpsilon=explorationProbEpsilon)
                rewardsQLearning[i] += reward
            #print('rewardsQLearning[i]', rewardsQLearning[i]) #AK
        rewardsQLearning /= num_runs
        if False:
            print('rewardsQLearning = ', rewardsQLearning)
            print('newStateActionValues = ', stateActionValues)
            # qlearning_policy = self.convert_action_values_into_policy(newStateActionValues)
            print('qlearning_policy = ', self.prettyPrintPolicy(stateActionValues))
        return stateActionValues, rewardsQLearning

def test_with_NextStateProbabilitiesEnv():
    S = 3
    A = 2
    nextStateProbability = np.random.rand(S,A,S) #positive numbers
    rewardsTable = np.random.randn(S,A,S) #can be negative
    environment = NextStateProbabilitiesEnv(nextStateProbability, rewardsTable)
    mdp = FiniteMDP(environment) #, S, A, discount=0.9)
    stateActionValues, rewardsQLearning = mdp.execute_q_learning(maxNumIterationsQLearning=1000)
    print("stateActionValues=",stateActionValues)
    print("rewardsQLearning=",rewardsQLearning)

def create_random_next_state_probability(S,A):
    nextStateProbability = np.random.rand(S,A,S) #positive numbers
    for s in range(S):
        for a in range(A):
            pmf = nextStateProbability[s,a] #probability mass function (pmf)
            total_prob = sum(pmf)
            if total_prob == 0:
                nextStateProbability[s,a,0] = 1 #arbitrarily, make first state have all probability
            else:
                nextStateProbability[s,a] /= total_prob #normalize to have a pmf
    return nextStateProbability

def test_with_sparse_NextStateProbabilitiesEnv():
    S = 3
    A = 2
    nextStateProbability = create_random_next_state_probability(S,A)
    #force some entries to be zero (we want to test the sparsity code)
    nextStateProbability[0,0] = np.zeros((S,))
    nextStateProbability[0,0,1] = 1
    rewardsTable = np.random.randn(S,A,S) #can be negative
    environment = NextStateProbabilitiesEnv(nextStateProbability, rewardsTable)
    mdp = FiniteMDP(environment) #, S, A, discount=0.9)
    valid_next_states = mdp.get_valid_next_states()
    state_values, iteration = mdp.compute_optimal_state_values()    
    print("state_values=",state_values)
    state_values, iteration = mdp.compute_optimal_state_values_sparse(valid_next_states)
    print("state_values with sparsity=",state_values)

    action_values, iteration = mdp.compute_optimal_action_values()
    print("action_values=",action_values)
    action_values, iteration = mdp.compute_optimal_action_values_sparse(valid_next_states)
    print("action_values with sparsity=",action_values)


if __name__ == '__main__':
    test_with_sparse_NextStateProbabilitiesEnv()
    #test_with_NextStateProbabilitiesEnv()

