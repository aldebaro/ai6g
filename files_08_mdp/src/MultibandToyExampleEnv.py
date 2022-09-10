'''
Esse dah suporte a FiniteMDP3.py

It is a subclass of NextStateProbabilitiesEnv.

This code implements the system (a toy example) described in slides for Diana.
It assumes knowledge of the correct nextStateProbability such that it allows to calculate optimum solution.
'''
import numpy as np
import itertools
import sys
#from akpy.FiniteMDP3 import FiniteMDP
from FiniteMDP import FiniteMDP
#from akpy.NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
from NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
import gym
from gym import spaces

class MultibandToyExampleEnv(NextStateProbabilitiesEnv):

    #def __init__(self, discount=1.0):
    def __init__(self):
        #call superclass constructor
        nextStateProbability, rewardsTable = self.createEnvironment()
        super().__init__(nextStateProbability, rewardsTable)
        ##NextStateProbabilitiesEnv.__init__(self,nextStateProbability, rewardsTable)
        
    def prettyPrint(self):
        '''Print MDP'''
        nextStateProbability = self.nextStateProbability
        rewardsTable = self.rewardsTable
        stateListGivenIndex = self.stateListGivenIndex
        actionListGivenIndex = self.actionListGivenIndex
        S = len(stateListGivenIndex)
        A = len(actionListGivenIndex)
        for s in range(S):
            currentState = stateListGivenIndex[s]
            print('current state s', s, '=', currentState, sep='')  # ' ',end='')
            for a in range(A):
                currentAction = actionListGivenIndex[a]
                print('  action a', a, '=', currentAction, sep='', end='')
                shouldPrintOnce = True
                for nexts in range(S):
                    if nextStateProbability[s, a, nexts] == 0:
                        continue
                        #nonZeroIndices = np.where(nextStateProbability[s, a] > 0)[0]
                    # if len(nonZeroIndices) != 2:
                    #    print('Error in logic, not 2: ', len(nonZeroIndices))
                    #    exit(-1)
                    r = rewardsTable[s, a, nexts]
                    newBuffers = stateListGivenIndex[nexts][0]
                    currentBuffers = currentState[0]
                    if r < 0 and r != self.outageReward:
                        drop = self.dictionaryOfUsersWithDroppedPackets[(s,a,nexts)]
                        extraPackets = np.array([0, 0, 0])
                        extraPackets[drop] = 1
                        transmitRate = np.array([1, 1, 1]) + currentBuffers - newBuffers -extraPackets
                    else:
                        transmitRate = np.array([1, 1, 1]) + currentBuffers - newBuffers
                    if shouldPrintOnce:
                        print(' transmit=', transmitRate)
                        shouldPrintOnce = False
                    if r < 0 and r != self.outageReward:
                        print('    next s', nexts, '=', stateListGivenIndex[nexts],
                              ' prob=', nextStateProbability[s, a, nexts],
                              ' reward=', r,
                              ' dropped=', drop,
                              sep='')
                    else:
                        print('    next s', nexts, '=', stateListGivenIndex[nexts],
                              ' prob=', nextStateProbability[s, a, nexts],
                              ' reward=', r,
                              sep='')

    def createActionsDataStructures(self, M, F):
        # dictionaries
        actionDictionaryGetIndex = dict()
        actionListGivenIndex = list()
        uniqueIndex = 0
        # AK-TODO could use itertools to get all combinations instead of assuming there are only 2 users and freqs.
        frequencies = ('H','L')  #need to have F elements
        for u1 in range(M):
            for u2 in range(u1, M):
                for f1 in range(F):
                    for f2 in range(F):
                        if u1 == u2:
                            continue
                        actionTuple = (u1, u2, frequencies[f1], frequencies[f2])
                        actionDictionaryGetIndex[actionTuple] = uniqueIndex
                        actionListGivenIndex.append(actionTuple)
                        uniqueIndex += 1
        return actionDictionaryGetIndex, actionListGivenIndex

    def createStatesDataStructures(self, M, B):
        '''M is the number of users and B the buffer size'''
        bufferStateList = list(itertools.product(np.arange(B + 1), repeat=M))
        N = len(bufferStateList)

        stateListGivenIndex = list()
        stateDictionaryGetIndex = dict()
        uniqueIndex = 0
        # add states without Bernoulli_extra_interference
        for i in range(N):
            onlyBuffersTuple = bufferStateList[i]
            augumentedTuple = (onlyBuffersTuple, 'N')
            stateListGivenIndex.append(augumentedTuple)
            stateDictionaryGetIndex[augumentedTuple] = uniqueIndex
            uniqueIndex += 1
        # add states with Bernoulli_extra_interference
        for i in range(N):
            onlyBuffersTuple = bufferStateList[i]
            augumentedTuple = (onlyBuffersTuple, 'I')
            stateListGivenIndex.append(augumentedTuple)
            stateDictionaryGetIndex[augumentedTuple] = uniqueIndex
            uniqueIndex += 1

        return stateDictionaryGetIndex, stateListGivenIndex

    def createEnvironment(self):
        '''
        We have S=16 states, A=12 possible actions and R=8 rewards, so we have 16 x 8 = 128 possible
        # state-reward pairs for each of the 16 x 12 = 192 possible current state-action pairs.
        In the most general case, we can represent the dynamics of the process with p(s',r | s,a).
            Obs: this can be codes as a 2d matrix with dimension 192 x 80 or as a 4d matrix with dimension S x A x S x R.
        But in our case, we have the reward being deterministic, not depending on next state. See:
        https://datascience.stackexchange.com/questions/17860/reward-dependent-on-state-action-versus-state-action-successor-state
        in other words: given a pair (s,a), we don't need a joint distribution over (s',r). We can simply describe
        the distribution p(s'|s,a) and have our function with expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        another matrix with the same dimension.
        '''
        # S = 16
        # A = 12
        self.outageReward = -5

        possibleRewards = [0, 1, 2, 3, 4, -10, -20, -30, self.outageReward]
        self.R = R = len(possibleRewards)
        self.B = B = 1  # buffer size
        self.alpha = alpha = 0.1  # from no to Bernoulli_extra_interference
        self.beta = beta = 0.4  # from Bernoulli_extra_interference to no

        self.M = M = 3 #number of users
        self.F = F = 2 #number of frequencies
        actionDictionaryGetIndex, actionListGivenIndex = self.createActionsDataStructures(M, F)
        A = len(actionListGivenIndex)

        self.dictionaryOfUsersWithDroppedPackets = dict()
        self.bitRates = np.zeros((self.M,))
        self.packetDropCounts = np.zeros((self.M,))

        if False:
            # print('AK = ', mydict)
            actionTuple = (1, 2, 0, 0)
            print('actionTuple = (1,2,0,0)')
            actionTupleIndex = actionDictionaryGetIndex[actionTuple]
            print('index = ', actionTupleIndex)
            (u1, u2, f1, f2) = actionListGivenIndex[actionTupleIndex]
            print('retrieved = ', (u1, u2, f1, f2))

        stateDictionaryGetIndex, stateListGivenIndex = self.createStatesDataStructures(M, B)
        S = len(stateListGivenIndex)

        #rewardDictionaryGetIndex, rewardListGivenIndex = self.createRewardsDataStructures(possibleRewards)

        # Assume ordering, such that 0 is always u1 (never u2) and 2 is always u2 (never u1)
        # print(actionListGivenIndex)
        # [(0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 1, 0), (0, 2, 1, 1), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 1, 0), (1, 2, 1, 1)]

        #now we need to populate the nextStateProbability array and rewardsTable
        # In the most general case, we can represent the dynamics of the process with p(s',r | s,a).
        # Obs: this can be coded e.g. with a 4D matrix with dimension S x R x S x A.
        # But in our case, we have the reward being deterministic, not depending on next state. See:
        # https://datascience.stackexchange.com/questions/17860/reward-dependent-on-state-action-versus-state-action-successor-state
        # in other words: given a pair (s,a), we don't need a joint distribution over (s',r). We can simply describe
        # the distribution p(s'|s,a) and have our function with expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        # In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        # another matrix with the same dimension.

        # jointProbNextStateAction = np.zeros((S, A, S, R)) #no need, see comments above
        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        plotBar = True
        for s in range(0, S):
            currentState = stateListGivenIndex[s]
            buffersTuple = currentState[0]
            currentInterference = False
            if currentState[1] == 'I':
                currentInterference = True
            for a in range(A):
                # for nexts in range(S):
                #    for r in range(R):
                currentAction = actionListGivenIndex[a]
                (u1, u2, f1, f2) = currentAction
                # start assuming maximum throughput
                t1 = 2
                t2 = 2
                if u2 == 2 and f2 == 'H':
                    t2 = 1  # user 2 is far from BS and get only 1 if higher freq. is used
                if currentInterference == True:
                    if u2 == 2:
                        t2 = 1  # with expected Bernoulli_extra_interference, user 2 only gets 1
                if u1 == 0 and u2 == 1:  # If MS0 and MS1 are scheduled simultaneously, the reward per MS is 1
                    t1 = 1
                    t2 = 1
                if f1 == f2:  # If two frequencies are the same, the rate per MS decreases by 1
                    t1 = np.max(t1 - 1, 0)
                    t2 = np.max(t2 - 1, 0)
                # need to reduce the rate in case there are fewer packets to send
                if t1 > buffersTuple[u1] + 1:
                    t1 = buffersTuple[u1] + 1
                if t2 > buffersTuple[u2] + 1:
                    t2 = buffersTuple[u2] + 1
                # predict the case of unexpected Bernoulli_extra_interference: it was false and went to green
                # AK-TODO will not implement now
                # if currentInterference == False:

                # new buffer state
                transmitRate = np.array([0, 0, 0])
                transmitRate[u1] = t1
                transmitRate[u2] = t2
                buffersArray = np.array(buffersTuple)
                newBuffers = np.array([1, 1, 1]) + buffersArray - transmitRate
                # drop = np.argwhere(newBuffers == 2)
                drop = newBuffers == 2
                newBuffers[drop] = 1
                newBuffersTuple = tuple(newBuffers)
                # calculate rewards
                sumDrops = np.sum(drop)
                if sumDrops > 0:
                    r = -10 * sumDrops
                else:
                    r = np.sum(transmitRate)

                if False: #print MDP (for debugging, see prettyPrint in superclass)
                    if plotBar == True:
                        print('------------------------------------------------------------------------------------------')
                    plotBar = not plotBar
                    print('s=', buffersTuple, ' a', a, '=', currentAction, ' tx=', transmitRate, ' nexts=',
                          newBuffersTuple, ' drop=', drop, ' r=', r, sep='')

                # probabilistic part: consider the two cases of Bernoulli_extra_interference or not
                #rewardIndice = rewardDictionaryGetIndex[r]  # just to check if dictionary is complete
                if currentInterference == True:
                    # stay within Bernoulli_extra_interference state
                    nextState = (newBuffersTuple, 'I')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = 1 - beta
                    rewardsTable[s, a, nextStateIndice] = r
                    #memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(s,a,nextStateIndice)]=drop
                    # go from Bernoulli_extra_interference to no int. state
                    nextState = (newBuffersTuple, 'N')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = beta
                    rewardsTable[s, a, nextStateIndice] = r
                    #memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(s,a,nextStateIndice)]=drop
                else:
                    # stay within no Bernoulli_extra_interference state
                    nextState = (newBuffersTuple, 'N')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = 1 - alpha
                    rewardsTable[s, a, nextStateIndice] = r
                    #memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(s,a,nextStateIndice)]=drop
                    # go from Bernoulli_extra_interference to no int. state
                    # in this case the reward is the minimum between the "drop" penalty and outage value
                    r = np.minimum(r, self.outageReward)
                    nextState = (newBuffersTuple, 'I')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = alpha
                    rewardsTable[s, a, nextStateIndice] = r
                    #memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(s,a,nextStateIndice)]=drop

        #self.prettyPrint(nextStateProbability, rewardsTable, stateListGivenIndex, actionListGivenIndex)

        self.actionDictionaryGetIndex = actionDictionaryGetIndex
        self.actionListGivenIndex = actionListGivenIndex
        self.stateDictionaryGetIndex = stateDictionaryGetIndex
        self.stateListGivenIndex = stateListGivenIndex
        #self.rewardDictionaryGetIndex = rewardDictionaryGetIndex
        #self.rewardListGivenIndex = rewardListGivenIndex
        #self.nextStateProbability = nextStateProbability
        #self.rewardsTable = rewardsTable
        #self.environment = NextStateProbabilitiesEnv(nextStateProbability, rewardsTable)

        return nextStateProbability, rewardsTable

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        currentIteration = history["time"]
        s= history["state_t"]
        a = history["action_t"]
        reward = history["reward_tp1"]
        nexts = history["state_tp1"]

        currentState = self.stateListGivenIndex[s]
        nextState = self.stateListGivenIndex[nexts]
        currentBuffer = np.array(nextState[0])
        nextBuffer = np.array(currentState[0])

        transmittedPackets = np.array([1,1,1])+currentBuffer-nextBuffer
        if reward < 0: #there was packet drop
            drop=self.dictionaryOfUsersWithDroppedPackets[(s,a,nexts)]
            transmittedPackets[drop] += -1
            droppedPackets = np.zeros((self.M,))
            droppedPackets[drop] = 1
            self.packetDropCounts += droppedPackets
        #compute bit rate
        self.bitRates += transmittedPackets
        if printPostProcessingInfo:
            print(self.bitRates, self.packetDropCounts)
        #print('accumulated rates =', self.bitRates, ' drops =', self.packetDropCounts)
        #print('kkkk = ', s, action, reward, nexts)

    def resetCounters(self):
        #reset counters
        self.bitRates = np.zeros((self.M,))
        self.packetDropCounts = np.zeros((self.M,))

def run_all():
    #mdp = FiniteMDP(discount=0.9)
    shouldPrintAll = True
    env = MultibandToyExampleEnv()
    mdp = FiniteMDP(env)

    #nextStateProbability, rewardsTable = createFiniteMDP()
    # nextStateProbability, rewardsTable = get_mdp_for_grid_world()
    # nextStateProbability, rewardsTable = get_simple_mdp()
    equiprobable_policy = mdp.getEquiprobableRandomPolicy()
    state_values, iteration = mdp.compute_state_values(equiprobable_policy)

    if shouldPrintAll:
        print('In-place:')
        print('State values under random policy after %d iterations', (iteration))
        print(state_values)

    state_values, iteration = mdp.compute_state_values(equiprobable_policy)
    if shouldPrintAll:
        print('Synchronous:')
        print('State values under random policy after %d iterations', (iteration))
        print(state_values)

    state_values, iteration = mdp.compute_optimal_state_values()
    if shouldPrintAll:
        print('Optimum states, iteration = ', iteration, ' state_values = ', np.round(state_values, 1))

    optimal_action_values, iteration = mdp.compute_optimal_action_values()
    if shouldPrintAll:
        print('Optimum actions, iteration = ', iteration, ' action_values = ', np.round(optimal_action_values, 1))

    #AK-TODO no need to be class method
    optimal_policy = mdp.convert_action_values_into_policy(optimal_action_values)
    if shouldPrintAll:
        print('policy = ', optimal_policy)
        mdp.prettyPrintValues(optimal_policy, env.stateListGivenIndex, env.actionListGivenIndex)

    q_learning_policy, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=100)

    #print('Example: ')
    if False:
        mdp.run_MDP_for_given_policy(optimal_policy, maxNumIterations=100)
    else:
        mdp.run_MDP_for_given_policy(equiprobable_policy,maxNumIterations=100)

def evaluateLearning():
    env = MultibandToyExampleEnv()
    mdp = FiniteMDP(env)
    alphas = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    epsilons = (0.1, 0.01, 0.001)
    for a in alphas:
        for e in epsilons:
            #print('alpha=',a,'epsilon=',e)
            fileName = 'smooth_q_eps' + str(e) + '_alpha' + str(a) + '.txt'
            sys.stdout = open(fileName, 'w')
            action_values, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=1000, maxNumIterationsQLearning = 1,
                                                                     num_runs = 20, stepSizeAlpha=a, explorationProbEpsilon=e)
            for i in range(len(rewardsQLearning)):
                print(rewardsQLearning[i])

def compareOptimalAndQLearningPolicies():

    env = MultibandToyExampleEnv()
    mdp = FiniteMDP(env)
    N=10000
    maxNumIterations=100
    statsQLearning = list()
    statsOptimal = list()
    for i in range(N):
        print(i)
        action_valuesOptimal, iteration = mdp.compute_optimal_action_values()
        action_valuesQLearning, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=400, maxNumIterationsQLearning = 1,
                                                                          num_runs = 1, stepSizeAlpha=0.5, explorationProbEpsilon=0.001)
        policyOptimal = mdp.convert_action_values_into_policy(action_valuesOptimal)
        policyQ = mdp.convert_action_values_into_policy(action_valuesQLearning)
        #to get performance while testing only
        env.resetCounters()
        print('Optimal')
        mdp.run_MDP_for_given_policy(policyOptimal,maxNumIterations=maxNumIterations,printPostProcessingInfo=False)
        statsOptimal.append( (np.sum(env.bitRates),  np.sum(env.packetDropCounts)) )
        env.resetCounters()
        print('Q-learning')
        mdp.run_MDP_for_given_policy(policyQ,maxNumIterations=maxNumIterations,printPostProcessingInfo=False)
        statsQLearning.append( (np.sum(env.bitRates)/maxNumIterations,  np.sum(env.packetDropCounts)/maxNumIterations) )

    sys.stdout = open('optimal.txt', 'w')
    for values in statsOptimal:
        print(values[0], values[1])

    sys.stdout = open('qlearning.txt', 'w')
    for values in statsQLearning:
        print(values[0], values[1])

if __name__ == '__main__':
    run_all()
    environment = MultibandToyExampleEnv()
    environment.prettyPrint()
    mdp = FiniteMDP(environment)
    #evaluateLearning()
    compareOptimalAndQLearningPolicies()

    #compareOptimalAndQLearningPolicies()
    #exit(1)
    
    print('AK')
    #mdp = MultibandFiniteMDPToyExample(discount=0.9)
    #mdp.prettyPrint()
    exit(1)
    if False:
        action_values, iteration = mdp.compute_optimal_action_values()
    else:
        action_values, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=1000, maxNumIterationsQLearning = 1,
                                                                 num_runs = 1000, stepSizeAlpha=0.9, explorationProbEpsilon=0.1)
        if False: #to get performance while training
            for i in range(len(rewardsQLearning)):
                print(rewardsQLearning[i])
        exit(1)

    if False:
        policy = action_values
    else:
        policy = mdp.convert_action_values_into_policy(action_values)

    #to get performance while testing only
    while True:
        mdp.run_MDP_for_given_policy(policy,maxNumIterations=10,printInfo=False, printPostProcessingInfo=False)

    #mdp.prettyPrintPolicy(policy)