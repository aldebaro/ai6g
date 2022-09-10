'''
Implements a simple user scheduling environment consisting of a finite MDP.
There are Nu=2 users.
The user positions are numbered from top-left to bottom-right in
zigzag scan. Assuming G=6:
(0,0) (0,1) ... (0,5)
(1,0) (1,1) ... (1,5)
...
(5,0) (5,1) ... (5,5)

It is a subclass of NextStateProbabilitiesEnv.

It assumes knowledge of the correct nextStateProbability such that it allows
to calculate optimum solution.
'''
from logging.config import valid_ident
import numpy as np
import itertools
import sys
#from akpy.FiniteMDP3 import FiniteMDP
from FiniteMDP import FiniteMDP
#from akpy.NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
from NextStateProbabilitiesEnv import NextStateProbabilitiesEnv
import gym
from gym import spaces

class UserSchedulingEnv(NextStateProbabilitiesEnv):

    #def __init__(self, discount=1.0):
    def __init__(self):
        # self.channel_spectral_efficiencies = channel_spectral_efficiencies
        #call superclass constructor
        self.ues_pos_prob, self.channel_spectral_efficiencies, self.ues_valid_actions = self.read_external_files()
        self.actions_move = np.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0],
        ])  # Up, right, down, left, stay
        nextStateProbability, rewardsTable = self.createEnvironment()
        super().__init__(nextStateProbability, rewardsTable)
        ##NextStateProbabilitiesEnv.__init__(self,nextStateProbability, rewardsTable)

    def read_external_files(self):
        ue0_file = np.load("./src/mobility_ue0.npz")
        ue0 = ue0_file.f.matrix_pos_prob
        ue0_valid = ue0_file.f.pos_actions_prob
        ue1_file = np.load("./src/mobility_ue1.npz")
        ue1 = ue1_file.f.matrix_pos_prob
        ue1_valid = ue1_file.f.pos_actions_prob
        capacity = np.load("./src/spec_eff_matrix.npz")
        capacity = capacity.f.spec_eff_matrix


        return ([ue0,ue1],capacity, [ue0_valid,ue1_valid])

    """
    Assumes Nu=2 users.
    """
    def prettyPrint(self):
        '''Print MDP'''
        nextStateProbability = self.nextStateProbability
        rewardsTable = self.rewardsTable
        stateGivenIndexList = self.stateGivenIndexList
        actionGivenIndexList = self.actionGivenIndexList
        S = len(stateGivenIndexList)
        A = len(actionGivenIndexList)
        for s in range(S):
            currentState = stateGivenIndexList[s]
            all_positions = currentState[0]
            buffers = currentState[1]
            #print('current state s', s, '=', currentState, sep='')  # ' ',end='')
            print('current state s', s, '= p=', all_positions, "b=", buffers, sep='')  # ' ',end='')
            for a in range(A):
                currentAction = actionGivenIndexList[a]
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
                    newBuffers = stateGivenIndexList[nexts][0]
                    currentBuffers = currentState[0]

    def createEnvironment(self):
        '''
        The state is defined as the position of the two users and their buffers occupancy.
        The action is to select one of the two users.
        Given that Nu=2 and G=6, there are S=xx states and A=2 actions
        The channel is fixed.
        The base station (BS) is located at the bottom-left corner.
        '''
        G = 6 #grid dimension
        B = 3  # buffer size
        Nu = 2 #number of users
        num_incoming_packets_per_time_slot = 2

        indexGivenActionDictionary, actionGivenIndexList = createActionsDataStructures()
        A = len(actionGivenIndexList)

        indexGivenStateDictionary, stateGivenIndexList = createStatesDataStructures()
        S = len(stateGivenIndexList)

        # np.savez_compressed("states_actions.npz", indexGivenStateDictionary=indexGivenStateDictionary, stateGivenIndexList=stateGivenIndexList, indexGivenActionDictionary=indexGivenActionDictionary, actionGivenIndexList=actionGivenIndexList)

        #now we need to populate the nextStateProbability array and rewardsTable,
        # the distribution p(s'|s,a) and expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        # In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        # another matrix with the same dimension.

        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        for s in range(S):
            #current state:
            currentState = stateGivenIndexList[s]
            (all_positions, buffers_occupancy) = currentState #interpret the state
            # print('Reading state: positions=', all_positions,'buffers=',buffers_occupancy)
            for a in range(A):                
                currentAction = actionGivenIndexList[a]
                chosen_user = a #in this case, the action is the user
                #get the channels spectral efficiency (SE)
                chosen_user_position = all_positions[chosen_user]

                se = self.channel_spectral_efficiencies[chosen_user_position[0],chosen_user_position[1]]
                #based on selected (chosen) user, update its buffer
                transmitRate = se #transmitted packets 

                new_buffer = np.array(buffers_occupancy)
                new_buffer[chosen_user] -= transmitRate #decrement buffer of chosen user
                new_buffer[new_buffer<0] = 0
                new_buffer += num_incoming_packets_per_time_slot #arrival of new packets

                #check if overflow
                #in case positive, limit the buffers to maximum capacity
                number_dropped_packets = new_buffer - B
                number_dropped_packets[number_dropped_packets<0] = 0

                #saturate buffer levels
                new_buffer -= number_dropped_packets

                buffers_occupancy=tuple(new_buffer) #convert to tuple to compose state

                # calculate rewards
                sumDrops = np.sum(number_dropped_packets)
                r = -sumDrops

                for ue1_action in np.arange(5):
                    for ue2_action in np.arange(5):
                        prob_ue1_action = self.ues_valid_actions[0][all_positions[0][0], all_positions[0][1]][ue1_action]
                        prob_ue2_action = self.ues_valid_actions[1][all_positions[1][0], all_positions[1][1]][ue2_action]
                        if prob_ue1_action!=0 and prob_ue2_action!=0:
                            #calculate nextState
                            new_position_ue1 = np.array(all_positions[0]) + self.actions_move[ue1_action]
                            new_position_ue2 = np.array(all_positions[1]) + self.actions_move[ue2_action]
                            # if not (np.array_equal(new_position_ue1, new_position_ue2)) and not np.array_equal(new_position_ue1, np.array([5,0])) and not np.array_equal(new_position_ue2, np.array([5,0])):
                            new_position = ((new_position_ue1[0],new_position_ue1[1]), (new_position_ue2[0],new_position_ue2[1]))
                            nextState = (new_position, buffers_occupancy)

                            # probabilistic part: consider the user mobility
                            nextStateIndice = indexGivenStateDictionary[nextState]
                            #take in account mobility
                            nextStateProbability[s, a, nextStateIndice] = prob_ue1_action * prob_ue2_action
                            rewardsTable[s, a, nextStateIndice] = r
        self.indexGivenActionDictionary = indexGivenActionDictionary
        self.actionGivenIndexList = actionGivenIndexList
        self.indexGivenStateDictionary = indexGivenStateDictionary
        self.stateGivenIndexList = stateGivenIndexList

        return nextStateProbability, rewardsTable

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        currentIteration = history["time"]
        s= history["state_t"]
        a = history["action_t"]
        reward = history["reward_tp1"]
        nexts = history["state_tp1"]

        currentState = self.stateGivenIndexList[s]
        nextState = self.stateGivenIndexList[nexts]
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
    env = UserSchedulingEnv()
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
        mdp.prettyPrintValues(optimal_policy, env.stateGivenIndexList, env.actionGivenIndexList)

    q_learning_policy, rewardsQLearning = mdp.execute_q_learning(maxNumIterations=100)

    #print('Example: ')
    if False:
        mdp.run_MDP_for_given_policy(optimal_policy, maxNumIterations=100)
    else:
        mdp.run_MDP_for_given_policy(equiprobable_policy,maxNumIterations=100)

def evaluateLearning():
    env = UserSchedulingEnv()
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

    env = UserSchedulingEnv()
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

"""
    Assumes Nu=2 users.
"""
def createActionsDataStructures():
            possibleActions = ['u1', 'u2']
            dictionaryGetIndex = dict()
            listGivenIndex = list()
            for uniqueIndex in range(len(possibleActions)):
                dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
                listGivenIndex.append(possibleActions[uniqueIndex])
            return dictionaryGetIndex, listGivenIndex


def createStatesDataStructures(G=6, Nu=2, B=3):
        show_debug_info = True
        # G is the axis dimension, on both horizontal and vertical
        # I cannot use below:
        #all_positions_list = list(itertools.product(np.arange(G), repeat=2))
        #because the base station is at position (G-1, 0) and users cannot be
        #in the same position at the same time        
        if show_debug_info:
            print("theoretical S=", (G**2)*(G**2)*((B+1)**Nu))

        # bs_position = (G-1, 0) #Base station position

        all_positions_of_single_user = list()
        for i in range(G):
            for j in range(G):
                # if (i == bs_position[0]) and (j == bs_position[1]):
                    # continue #do not allow user at the BS position
                all_positions_of_single_user.append((i,j))

        #create Cartesian product among positions of users
        all_positions_list = list(itertools.product(all_positions_of_single_user, repeat=Nu))
        #need to remove the positions that coincide
        N = len(all_positions_list)
        #print("N=",N)
        # i = 0
        # while (True):
        #     positions_pair = all_positions_list[i]
        #     position_u1 = positions_pair[0]
        #     position_u2 = positions_pair[1]
        #     if position_u1 == position_u2:
        #         all_positions_list.pop(i)
        #         N -= 1 #decrement the number N of elements in list
        #     else:
        #         i += 1 #continue searching the list
        #     if i >= N:
        #         break

        all_buffer_occupancy_list = list(itertools.product(np.arange(B + 1), repeat=Nu))
        all_states = itertools.product(all_positions_list, all_buffer_occupancy_list)
        all_states = list(all_states)
        # if show_debug_info:
        #     print("all_positions_list",all_positions_list)
        #     #Nu is the number of users and B the buffer size
        #     print("all_buffer_occupancy_list",all_buffer_occupancy_list)
        #     print('num of position states=', len(all_positions_list))
        #     print('num of buffer states=', len(all_buffer_occupancy_list))
        #     print("calculated total num S of states= ", len(all_states))

        N = len(all_states) #number of states
        stateGivenIndexList = list()
        indexGivenStateDictionary = dict()
        uniqueIndex = 0
        # add states to both dictionary and its inverse mapping (a list)
        for i in range(N):
            stateGivenIndexList.append(all_states[i])
            indexGivenStateDictionary[all_states[i]] = uniqueIndex
            uniqueIndex += 1
        if False:
            print('indexGivenStateDictionary = ', indexGivenStateDictionary)
            print('stateGivenIndexList = ', stateGivenIndexList)
        return indexGivenStateDictionary, stateGivenIndexList

def check_matrix(nextState):
    a = np.sum(nextState) == nextState.shape[0]
    print(a)

if __name__ == '__main__':
    G=6
    se = np.ones((G,G))
    env = UserSchedulingEnv()
    # env.prettyPrint()
    Nu=2
    B=3
    check_matrix(env.nextStateProbability)
    createStatesDataStructures(G, Nu, B)

    if False:
        run_all()
        environment = UserSchedulingEnv()
        environment.prettyPrint()
        mdp = FiniteMDP(environment)
        #evaluateLearning()
        compareOptimalAndQLearningPolicies()
