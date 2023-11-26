'''
Implements a simple user scheduling environment consisting of a finite MDP.
The user positions are numbered from top-left to bottom-right in
zigzag scan. Assuming G=6:
(0,0) (0,1) ... (0,5)
(1,0) (1,1) ... (1,5)
...
(5,0) (5,1) ... (5,5)

 In some grid worlds, the agent action is where to move, but here the
 users' mobility is an external process and the agent simply needs to
 schedule (choose) the users. There are Nu=2 users and, consequently Nu actions.

The state is defined as the position of the two users and their buffers occupancy.
There are S=(G**2)*(G**2)*((B+1)**Nu) states, where G is the grid dimension, B is the
buffer size, and Nu is the number of user, which coincides with number of actions.
With G=6, B=3 and Nu=2, the number S of states is 20736.

The channels are considered fixed (do not vary over time), and depend only
on the grid position. Therefore, the channel spectral efficiency is provided
by a G x G matrix, which is valid for all users. This spectral efficiency is
provided in number of packets that can be transmitted per channel use, in a
given time slot.
'''
import numpy as np
import itertools
import pickle
import os

from known_dynamics_env import KnownDynamicsEnv
import finite_mdp_utils as fmdp


class UserSchedulingEnv(KnownDynamicsEnv):

    def __init__(self):
        self.ues_pos_prob, self.channel_spectral_efficiencies, self.ues_valid_actions = self.read_external_files()

        self.channel_spectral_efficiencies = get_channel_spectral_efficiency()

        print("self.channel_spectral_efficiencies",
              self.channel_spectral_efficiencies)
        # print(self.ues_valid_actions[0])
        # print(len(self.ues_pos_prob[1][4]))

        self.actions_move = np.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0],
        ])  # Up, right, down, left, stay

        nextStateProbability, rewardsTable, actionDictionaryGetIndex, actionListGivenIndex, stateDictionaryGetIndex, stateListGivenIndex = self.createEnvironment()

        # pack data structures (dic and list) to map names into indices for actions
        actions_info = [actionDictionaryGetIndex, actionListGivenIndex]

        # pack data structures (dic and list) to map names into indices for states
        states_info = [stateDictionaryGetIndex, stateListGivenIndex]

        # call superclass constructor
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable,
                                  actions_info=actions_info, states_info=states_info)
        print("INFO: finished creating environment")

    def read_external_files(self):
        ue0_file = np.load("../mobility_ue0.npz")
        ue0 = ue0_file.f.matrix_pos_prob
        ue0_valid = ue0_file.f.pos_actions_prob
        ue1_file = np.load("../mobility_ue1.npz")
        ue1 = ue1_file.f.matrix_pos_prob
        ue1_valid = ue1_file.f.pos_actions_prob
        capacity = np.load("../spec_eff_matrix.npz")
        capacity = capacity.f.spec_eff_matrix

        return ([ue0, ue1], capacity, [ue0_valid, ue1_valid])

    def prettyPrint(self):
        """
        Print MDP. Assumes Nu=2 users.
        """
        nextStateProbability = self.nextStateProbability
        rewardsTable = self.rewardsTable
        stateListGivenIndex = self.stateListGivenIndex
        actionListGivenIndex = self.actionListGivenIndex
        S = len(stateListGivenIndex)
        A = len(actionListGivenIndex)
        for s in range(S):
            currentState = stateListGivenIndex[s]
            all_positions = currentState[0]
            buffers = currentState[1]
            # print('current state s', s, '=', currentState, sep='')  # ' ',end='')
            print('current state s', s, '= p=', all_positions,
                  "b=", buffers, sep='')  # ' ',end='')
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

    def createEnvironment(self):
        '''
        The state is defined as the position of the two users and their buffers occupancy.
        The action is to select one of the two users.
        Given that Nu=2 and G=6, there are S=20736 states and A=2 actions
        The wireless channel is fixed: each grid point has a given channel that does not change.
        The base station (BS) is located at the bottom-left corner.
        '''
        G = 6  # grid dimension
        B = 3  # buffer size
        Nu = 2  # number of users
        num_incoming_packets_per_time_slot = 2

        actionDictionaryGetIndex, actionListGivenIndex = createActionsDataStructures()
        A = len(actionListGivenIndex)

        stateDictionaryGetIndex, stateListGivenIndex = createStatesDataStructures()
        S = len(stateListGivenIndex)

        # np.savez_compressed("states_actions.npz", stateDictionaryGetIndex=stateDictionaryGetIndex, stateListGivenIndex=stateListGivenIndex, actionDictionaryGetIndex=actionDictionaryGetIndex, actionListGivenIndex=actionListGivenIndex)

        # now we need to populate the nextStateProbability array and rewardsTable,
        # the distribution p(s'|s,a) and expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        # In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        # another matrix with the same dimension.
        nextStateProbability = np.zeros((S, A, S))
        rewardsTable = np.zeros((S, A, S))
        for s in range(S):
            # current state:
            currentState = stateListGivenIndex[s]
            (all_positions, buffers_occupancy) = currentState  # interpret the state
            # print('Reading state: positions=', all_positions,'buffers=',buffers_occupancy)
            for a in range(A):
                currentAction = actionListGivenIndex[a]
                chosen_user = a  # in this case, the action is the user
                # get the channels spectral efficiency (SE)
                chosen_user_position = all_positions[chosen_user]

                se = self.channel_spectral_efficiencies[chosen_user_position[0],
                                                        chosen_user_position[1]]
                # based on selected (chosen) user, update its buffer
                transmitRate = se  # transmitted packets

                new_buffer = np.array(buffers_occupancy)
                # decrement buffer of chosen user
                new_buffer[chosen_user] -= transmitRate
                new_buffer[new_buffer < 0] = 0
                new_buffer += num_incoming_packets_per_time_slot  # arrival of new packets

                # check if overflow
                # in case positive, limit the buffers to maximum capacity
                number_dropped_packets = new_buffer - B
                number_dropped_packets[number_dropped_packets < 0] = 0

                # saturate buffer levels
                new_buffer -= number_dropped_packets

                # convert to tuple to compose state
                buffers_occupancy = tuple(new_buffer)

                # calculate rewards
                sumDrops = np.sum(number_dropped_packets)
                r = -sumDrops

                for ue1_action in np.arange(5):
                    for ue2_action in np.arange(5):
                        prob_ue1_action = self.ues_valid_actions[0][all_positions[0]
                                                                    [0], all_positions[0][1]][ue1_action]
                        prob_ue2_action = self.ues_valid_actions[1][all_positions[1]
                                                                    [0], all_positions[1][1]][ue2_action]
                        if prob_ue1_action != 0 and prob_ue2_action != 0:
                            # calculate nextState
                            new_position_ue1 = np.array(
                                all_positions[0]) + self.actions_move[ue1_action]
                            new_position_ue2 = np.array(
                                all_positions[1]) + self.actions_move[ue2_action]
                            # if not (np.array_equal(new_position_ue1, new_position_ue2)) and not np.array_equal(new_position_ue1, np.array([5,0])) and not np.array_equal(new_position_ue2, np.array([5,0])):
                            new_position = (
                                (new_position_ue1[0], new_position_ue1[1]), (new_position_ue2[0], new_position_ue2[1]))
                            nextState = (new_position, buffers_occupancy)

                            # probabilistic part: consider the user mobility
                            nextStateIndice = stateDictionaryGetIndex[nextState]
                            # /take in account mobility
                            nextStateProbability[s, a,
                                                 nextStateIndice] = prob_ue1_action * prob_ue2_action
                            rewardsTable[s, a, nextStateIndice] = r

        return nextStateProbability, rewardsTable, actionDictionaryGetIndex, actionListGivenIndex, stateDictionaryGetIndex, stateListGivenIndex

    def TODO_postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results
        AK-TODO: this was copied from multiband_scheduling
        but was not finished. I am not sure we need it.
        For now, I will use the superclass method, which is a pass'''
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

        # state is something as (((0, 0), (3, 0)), (3, 2))
        # two positions and buffer occupancy
        currentBuffer = np.array(nextState[1])
        nextBuffer = np.array(currentState[1])

        transmittedPackets = np.array([1, 1])+currentBuffer-nextBuffer
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


def createActionsDataStructures():
    """
        Assumes Nu=2 users.
    """
    possibleActions = ['u1', 'u2']
    dictionaryGetIndex = dict()
    listGivenIndex = list()
    for uniqueIndex in range(len(possibleActions)):
        dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
        listGivenIndex.append(possibleActions[uniqueIndex])
    return dictionaryGetIndex, listGivenIndex


def get_channel_spectral_efficiency(G=6, ceil_value=5) -> np.ndarray:
    '''
    Create spectral efficiency as 0, 0, ..., 0, 1, 2, 3
    '''
    if ceil_value > G*G-1:
        raise Exception(
            "Decrease ceil_value otherwise spectral efficiencies are all zero")
    channel_spectral_efficiencies = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            if i+j > ceil_value:
                channel_spectral_efficiencies[i, j] = 0
            else:
                channel_spectral_efficiencies[i, j] = i+j
    return channel_spectral_efficiencies


def createStatesDataStructures(G=6, Nu=2, B=3):
    show_debug_info = False
    # G is the axis dimension, on both horizontal and vertical
    # I cannot use below:
    # all_positions_list = list(itertools.product(np.arange(G), repeat=2))
    # because the base station is at position (G-1, 0) and users cannot be
    # in the same position at the same time
    if show_debug_info:
        print("theoretical S=", (G**2)*(G**2)*((B+1)**Nu))

    # bs_position = (G-1, 0) #Base station position

    all_positions_of_single_user = list()
    for i in range(G):
        for j in range(G):
            # if (i == bs_position[0]) and (j == bs_position[1]):
            # continue #do not allow user at the BS position
            all_positions_of_single_user.append((i, j))

    # create Cartesian product among positions of users
    all_positions_list = list(itertools.product(
        all_positions_of_single_user, repeat=Nu))
    # need to remove the positions that coincide
    N = len(all_positions_list)
    # print("N=",N)
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

    all_buffer_occupancy_list = list(
        itertools.product(np.arange(B + 1), repeat=Nu))
    all_states = itertools.product(
        all_positions_list, all_buffer_occupancy_list)
    all_states = list(all_states)
    if show_debug_info:
        print("all_positions_list", all_positions_list)
        # Nu is the number of users and B the buffer size
        print("all_buffer_occupancy_list", all_buffer_occupancy_list)
        print('num of position states=', len(all_positions_list))
        print('num of buffer states=', len(all_buffer_occupancy_list))
        print("calculated total num S of states= ", len(all_states))

    N = len(all_states)  # number of states
    stateListGivenIndex = list()
    stateDictionaryGetIndex = dict()
    uniqueIndex = 0
    # add states to both dictionary and its inverse mapping (a list)
    for i in range(N):
        stateListGivenIndex.append(all_states[i])
        stateDictionaryGetIndex[all_states[i]] = uniqueIndex
        uniqueIndex += 1
    if False:
        print('stateDictionaryGetIndex = ', stateDictionaryGetIndex)
        print('stateListGivenIndex = ', stateListGivenIndex)
    return stateDictionaryGetIndex, stateListGivenIndex


def check_matrix(nextStateProbability):
    '''
    Sanity chech of nextStateProbability
    '''
    num_states, num_actions, num_states2 = nextStateProbability.shape
    assert num_states == num_states2
    assert num_actions == 2
    # there are 2 actions, each one corresponding to a probability distribution that sums up to 1
    assert num_states * num_actions == np.sum(nextStateProbability)
    if False:
        print(nextStateProbability)
        print(np.sum(nextStateProbability))
        print(nextStateProbability.shape)


if __name__ == '__main__':
    print("Creating the UserSchedulingEnv environment... It takes some time.")

    file_path = "UserSchedulingEnv.pickle"
    if os.path.exists(file_path):
        print(f"The file {file_path} exists. I will read it")
        env = pickle.load(open(file_path, "rb"))
    else:
        print(
            f"The file {file_path} does not exist. I will create the object and create the file")
        env = UserSchedulingEnv()  # grid of size G = 6
        pickle.dump(env, open(file_path, "wb"))

    # env.prettyPrint()
    # check_matrix(env.nextStateProbability) # it was ok

    print("Example of action:", env.actionListGivenIndex[0])
    print("Example of state:", env.stateListGivenIndex[302])

    # try one step
    action = 0
    ob, reward, gameOver, history = env.step(action)
    printPostProcessingInfo = True
    env.postprocessing_MDP_step(history, printPostProcessingInfo)

    if True:
        fmdp.compare_q_learning_with_optimum_policy(env)
    if True:
        # this may take long time to run
        fmdp.hyperparameter_grid_search(env)
