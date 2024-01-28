'''
Multiband scheduling and frequency band allocation.
The documentation is at the companion set of slides: multiband_scheduling_env_documentation.pptx

It is a subclass of NextStateProbabilitiesEnv.
It assumes knowledge of the correct nextStateProbability such that it allows to calculate optimum solution.
'''
import numpy as np
import itertools
import sys
import gymnasium as gym
from gymnasium import spaces

from known_dynamics_env import KnownDynamicsEnv
import finite_mdp_utils as fmdp

class MultibandToyExampleEnv(KnownDynamicsEnv):

    # def __init__(self, discount=1.0):
    def __init__(self):
        self.__version__ = "0.1.0"
        nextStateProbability, rewardsTable, actionDictionaryGetIndex, actionListGivenIndex, stateDictionaryGetIndex, stateListGivenIndex = self.createEnvironment()

        # pack data structures (dic and list) to map names into indices for actions
        actions_info = [actionDictionaryGetIndex, actionListGivenIndex]

        # pack data structures (dic and list) to map names into indices for states
        states_info = [stateDictionaryGetIndex, stateListGivenIndex]

        # call superclass constructor
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable,
                                  actions_info=actions_info, states_info=states_info)

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
            print('current state s', s, '=',
                  currentState, sep='')  # ' ',end='')
            for a in range(A):
                currentAction = actionListGivenIndex[a]
                print('  action a', a, '=', currentAction, sep='', end='')
                shouldPrintOnce = True
                for nexts in range(S):
                    if nextStateProbability[s, a, nexts] == 0:
                        continue
                    r = rewardsTable[s, a, nexts]
                    newBuffers = stateListGivenIndex[nexts][0]
                    currentBuffers = currentState[0]
                    if r < 0:
                        drop = self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nexts)]
                        extraPackets = np.array([0, 0, 0])
                        extraPackets[drop] = 1
                        transmitRate = np.array(
                            [1, 1, 1]) + currentBuffers - newBuffers - extraPackets
                    else:
                        transmitRate = np.array(
                            [1, 1, 1]) + currentBuffers - newBuffers
                    if shouldPrintOnce:
                        print(' transmit=', transmitRate)
                        shouldPrintOnce = False
                    if r < 0:
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

    def prettyPrint_with_enabled_reward_due_transition_N_to_I(self):
        '''
        Print MDP. This method can be used when self.reward_due_transition_N_to_I
        is set to true. In this case, the reward depends not only on (s, a) but
        on the next state s' too, that is, it depends on (s, a, s')
        '''
        nextStateProbability = self.nextStateProbability
        rewardsTable = self.rewardsTable
        stateListGivenIndex = self.stateListGivenIndex
        actionListGivenIndex = self.actionListGivenIndex
        S = len(stateListGivenIndex)
        A = len(actionListGivenIndex)
        for s in range(S):
            currentState = stateListGivenIndex[s]
            print('current state s', s, '=',
                  currentState, sep='')  # ' ',end='')
            for a in range(A):
                currentAction = actionListGivenIndex[a]
                print('  action a', a, '=', currentAction, sep='', end='')
                shouldPrintOnce = True
                for nexts in range(S):
                    if nextStateProbability[s, a, nexts] == 0:
                        continue
                        # nonZeroIndices = np.where(nextStateProbability[s, a] > 0)[0]
                    # if len(nonZeroIndices) != 2:
                    #    print('Error in logic, not 2: ', len(nonZeroIndices))
                    #    exit(-1)
                    r = rewardsTable[s, a, nexts]
                    newBuffers = stateListGivenIndex[nexts][0]
                    currentBuffers = currentState[0]
                    if r < 0 and r != self.reward_due_transition_N_to_I:
                        drop = self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nexts)]
                        extraPackets = np.array([0, 0, 0])
                        extraPackets[drop] = 1
                        transmitRate = np.array(
                            [1, 1, 1]) + currentBuffers - newBuffers - extraPackets
                    else:
                        transmitRate = np.array(
                            [1, 1, 1]) + currentBuffers - newBuffers
                    if shouldPrintOnce:
                        print(' transmit=', transmitRate)
                        shouldPrintOnce = False
                    if r < 0 and r != self.reward_due_transition_N_to_I:
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
        '''
        Creates two data structures to facilitate dealing with actions: 
        a list and a dictionary.
        '''
        # dictionaries
        actionDictionaryGetIndex = dict()
        actionListGivenIndex = list()
        uniqueIndex = 0
        # we could use itertools to get all combinations.
        # instead we assume here that there are only 2 users and freqs.
        frequencies = ('H', 'L')  # need to have F elements
        for u1 in range(M):
            for u2 in range(u1, M):
                for f1 in range(F):
                    for f2 in range(F):
                        if u1 == u2:
                            continue
                        actionTuple = (
                            u1, u2, frequencies[f1], frequencies[f2])
                        actionDictionaryGetIndex[actionTuple] = uniqueIndex
                        actionListGivenIndex.append(actionTuple)
                        uniqueIndex += 1
        return actionDictionaryGetIndex, actionListGivenIndex

    def createStatesDataStructures(self, M, B):
        '''
        Creates two data structures to facilitate dealing with states: 
        a list and a dictionary.
        '''
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

    def createRewardsDataStructures(self, possibleRewards):
        '''
        Creates two data structures to facilitate dealing with rewards: 
        a list and a dictionary.
        '''
        # dictionaries
        rewardDictionaryGetIndex = dict()
        rewardListGivenIndex = list()
        R = len(possibleRewards)
        uniqueIndex = 0
        for i in range(R):
            r = possibleRewards[i]
            rewardDictionaryGetIndex[r] = uniqueIndex
            rewardListGivenIndex.append(r)
            uniqueIndex += 1
        return rewardDictionaryGetIndex, rewardListGivenIndex


    def createEnvironment(self):
        '''
        Most important function: creates the matrices that describe the MDP dynamics.
        We have S=16 states, A=12 possible actions and R=8 rewards.
        Hence, we have 16 x 8 = 128 possible state-reward pairs
        for each of the 16 x 12 = 192 possible current state-action pairs.
        In the most general case, we can represent the dynamics of the process with p(s',r |s,a),
        which can be coded as a 4d matrix with dimension S x A x S x R.
        But in our case, we have the reward being deterministic, not depending on next state. See:
        https://datascience.stackexchange.com/questions/17860/reward-dependent-on-state-action-versus-state-action-successor-state
        In other words: given a pair (s,a), we don't need a joint distribution over (s',r). We can simply describe
        the distribution p(s'|s,a) and have our function with expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        another matrix with the same dimension.

        Instead of using:
        jointProbNextStateAction = np.zeros((S, A, S, R)) #no need, see comments above
        we will use instead:
        nextStateProbability = np.zeros((S, A, S))

        We use r(s,a,s') because the superclass KnownDynamicsEnv requires it, but unless use_fancy_reward_definition is enabled
        the reward does not depend on s' and instead of r(s,a,s') we could use r(s,a).
        Hence, to comply with the superclass KnownDynamicsEnv we will use
        rewardsTable = np.zeros((S, A, S))
        instead of rewardsTable = np.zeros((S, A))
        '''

        debug_reward_values = True # in case you want to debug the reward values or check them
        # possible values for rewards, from sum rate to dropped packets multiplied by -10
        possibleRewards = [0, 1, 2, 3, 4, -10, -20, -30]

        # in our research we tried this fancier reward definition, which creates an
        # extra value of reward, when the system goes from no-interference to interference
        # with this enabled (use_fancy_reward_definition = True), the reward depends
        # on next state s'
        # you can disable this feature.
        use_fancy_reward_definition = False
        if use_fancy_reward_definition:
            self.reward_due_transition_N_to_I = -5
            possibleRewards = [0, 1, 2, 3, 4, -10, -20, -30, self.reward_due_transition_N_to_I]        
                
        self.R = R = len(possibleRewards)
        self.B = B = 1  # buffer size of 1 packet per user
        # probabilities that define the intercell interference represented as states: 'I' and 'N'
        self.alpha = alpha = 0.1  # transition prob from 'N' to 'I'. The loop prob. is 1-alpha
        self.beta = beta = 0.4  # transition prob from 'I' to 'N'. The loop prob. is 1-beta

        self.M = M = 3  # number of users
        self.F = F = 2  # number of frequencies

        # initialize variables and data structures
        actionDictionaryGetIndex, actionListGivenIndex = self.createActionsDataStructures(
            M, F)
        A = len(actionListGivenIndex)

        self.dictionaryOfUsersWithDroppedPackets = dict()
        self.bitRates = np.zeros((self.M,))
        self.packetDropCounts = np.zeros((self.M,))

        stateDictionaryGetIndex, stateListGivenIndex = self.createStatesDataStructures(
            M, B)
        S = len(stateListGivenIndex)

        if debug_reward_values:
            rewardDictionaryGetIndex, rewardListGivenIndex = self.createRewardsDataStructures(possibleRewards)

        # now we need to populate the nextStateProbability array and rewardsTable
        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        for s in range(0, S):
            currentState = stateListGivenIndex[s]
            buffersTuple = currentState[0]
            currentInterference = False
            if currentState[1] == 'I':
                currentInterference = True
            for a in range(A): # go over all pairs (s, a)
                currentAction = actionListGivenIndex[a]
                (u1, u2, f1, f2) = currentAction
                # start assuming maximum throughput of 2 packets per user
                t1 = 2
                t2 = 2
                if u2 == 2 and f2 == 'H':
                    t2 = 1  # user 2 is far from BS and get only 1 if higher freq. is used
                if currentInterference == True: # in this case 'I' there is intercell interference
                    # check whether u2 is scheduled in this TTI
                    if u2 == 2: # user 2 is never scheduled as u1, so we can check if u2==2
                        t2 = 1  # when system is experiencing intercell interference, user 2 can only transmit 1 packet
                if u1 == 0 and u2 == 1:  # If U0 and U1 are scheduled simultaneously, they transmit only 1 packet each
                    t1 = 1
                    t2 = 1
                if f1 == f2:  # If two frequencies are the same, the rate per MS decreases by 1
                    t1 = np.max(t1 - 1, 0)
                    t2 = np.max(t2 - 1, 0)
                # need to reduce the rate in case there are fewer packets to send
                # the +1 below represents the new packet that arrives at each new time instant
                if t1 > buffersTuple[u1] + 1:
                    t1 = buffersTuple[u1] + 1
                if t2 > buffersTuple[u2] + 1:
                    t2 = buffersTuple[u2] + 1

                # new buffer state. Note that it does not depend on interference
                transmitRate = np.array([0, 0, 0]) # initialize
                transmitRate[u1] = t1
                transmitRate[u2] = t2
                buffersArray = np.array(buffersTuple) # enable doing arithmetics
                newBuffers = np.array([1, 1, 1]) + buffersArray - transmitRate
                # instead of drop = np.argwhere(newBuffers == 2), use:
                drop = newBuffers == 2 # check if buffer overflow occurred (dropped packets)
                newBuffers[drop] = 1 # if buffer overflow happened, correct the buffer occupancy to its maximum of 1
                newBuffersTuple = tuple(newBuffers) # convert back from array to tuple

                # calculate initial reward value
                # (it may change in case use_fancy_reward_definition is True and
                #  a transition from 'N' to 'I' happens)
                sumDrops = np.sum(drop) # drop is array with number of dropped packets per user
                if sumDrops > 0:
                    r = -10 * sumDrops
                else:
                    r = np.sum(transmitRate) # transmitRate is array with number of successfully transmitted packets per user

                # probabilistic part: consider the two cases of intercell interference: 'I' and 'N'
                if currentInterference == True: # case 'I', there is interference
                    # a) assume that system continues to have interference (remains at 'I' state)
                    nextState = (newBuffersTuple, 'I')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = 1 - beta # loop probability to remain in 'I'
                    rewardsTable[s, a, nextStateIndice] = r
                    # memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nextStateIndice)] = drop
                    # b) assume that system makes a transition to no interference (jumps from 'I' to 'N' state)
                    nextState = (newBuffersTuple, 'N')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = beta # transition probability
                    rewardsTable[s, a, nextStateIndice] = r
                    # memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nextStateIndice)] = drop
                else: # case 'N', there is no interference at current time instant
                    # a) assume that system continues to have no interference (remains at 'N' state)
                    nextState = (newBuffersTuple, 'N')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = 1 - alpha # loop probability to remain in 'N'
                    rewardsTable[s, a, nextStateIndice] = r
                    # memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nextStateIndice)] = drop
                    # b) assume that system makes a transition to interference (jumps from 'N' to 'I' state)
                    if use_fancy_reward_definition:
                        # in this case 'N' to 'I', the reward is decreased:
                        r = np.minimum(r, self.reward_due_transition_N_to_I)
                    nextState = (newBuffersTuple, 'I')
                    nextStateIndice = stateDictionaryGetIndex[nextState]
                    nextStateProbability[s, a, nextStateIndice] = alpha # transition probability
                    rewardsTable[s, a, nextStateIndice] = r
                    # memorize which users had packets dropped
                    if r < 0:
                        self.dictionaryOfUsersWithDroppedPackets[(
                            s, a, nextStateIndice)] = drop

                if debug_reward_values: # to debug
                    rewardIndice = rewardDictionaryGetIndex[r]  # just to check if dictionary is complete

        return nextStateProbability, rewardsTable, actionDictionaryGetIndex, actionListGivenIndex, stateDictionaryGetIndex, stateListGivenIndex

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        currentIteration = history["time"]
        s = history["state_t"]
        a = history["action_t"]
        reward = history["reward_tp1"]
        nexts = history["state_tp1"]

        if isinstance(s, int):
            currentState = self.stateListGivenIndex[s]
            nextState = self.stateListGivenIndex[nexts]
        else:
            currentState = s
            nextState = nexts
            # s, a and nexts must be indices here
            s = self.stateDictionaryGetIndex[s]
            nexts = self.stateDictionaryGetIndex[nexts]
            a = self.actionDictionaryGetIndex[a]

        # state is something as s ((1, 1, 0), 'N')
        currentBuffer = np.array(nextState[0])
        nextBuffer = np.array(currentState[0])

        transmittedPackets = np.array([1, 1, 1])+currentBuffer-nextBuffer
        if reward < 0:  # there was packet drop
            drop = self.dictionaryOfUsersWithDroppedPackets[(s, a, nexts)]
            transmittedPackets[drop] += -1
            droppedPackets = np.zeros((self.M,))
            droppedPackets[drop] = 1
            self.packetDropCounts += droppedPackets
        # compute bit rate
        self.bitRates += transmittedPackets
        if printPostProcessingInfo:
            print(self.bitRates, self.packetDropCounts)
        # print('accumulated rates =', self.bitRates, ' drops =', self.packetDropCounts)
        # print('kkkk = ', s, action, reward, nexts)

    def resetCounters(self):
        # reset counters
        self.bitRates = np.zeros((self.M,))
        self.packetDropCounts = np.zeros((self.M,))


def evaluateLearning():
    env = MultibandToyExampleEnv()
    alphas = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    epsilons = (0.1, 0.01, 0.001)
    for a in alphas:
        for e in epsilons:
            # print('alpha=',a,'epsilon=',e)
            fileName = 'smooth_q_eps' + str(e) + '_alpha' + str(a) + '.txt'
            sys.stdout = open(fileName, 'w')
            action_values, rewardsQLearning = fmdp.q_learning_several_episodes(
                env, num_runs=20, stepSizeAlpha=a, explorationProbEpsilon=e)
            for i in range(len(rewardsQLearning)):
                print(rewardsQLearning[i])


if __name__ == '__main__':
    env = MultibandToyExampleEnv()
    env.prettyPrint()
    exit(1)
    #fmdp.compare_q_learning_with_optimum_policy(
    #    env, output_files_prefix="MultibandToyExample")
    # if True:
    # this may take long time to run
    # fmdp.hyperparameter_grid_search(env)

    # Get optimum action values
    action_values, stopping_criteria = fmdp.compute_optimal_action_values(
        env, tolerance=0)
    iteration = stopping_criteria.shape[0]  # number of iterations
    stopping_criterion = stopping_criteria[-1]  # final stopping criterion
    print("All values of stopping criteria along the iterations: ", stopping_criteria)

    # Now convert from values to policy, to show the optimum policy
    policy = fmdp.convert_action_values_into_policy(action_values)
    fmdp.pretty_print_policy(env, policy)
    print("\nConverged with", iteration,
          "iterations with final stopping criterion=", stopping_criterion)
