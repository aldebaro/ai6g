'''

This problem is described in the paper https://biblioteca.sbrt.org.br/articles/3686

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

@TODO - support rendering with pygame. Get already created code.
@TODO - obtain the SE based on the positions p1 and p2, now it is fixed and arbitrary
@TODO - Use the same nomenclature. Sometimes we use:
        stateDictionaryGetIndex, stateListGivenIndex
        and others:
        self.indexGivenStateDictionary, self.stateGivenIndexList
        The same for actions.

'''
import numpy as np
import itertools
import pickle
import os

from known_dynamics_env import KnownDynamicsEnv
import finite_mdp_utils as fmdp
from mobility_utils import all_valid_next_moves, combined_users_positions, one_step_moves_in_grid


class UserSchedulingEnv(KnownDynamicsEnv):

    def __init__(self, G=6, B=3, Nu=2, num_pkts_per_tti=2,
                 can_users_share_position=False,
                 should_add_not_moving=False,
                 print_debug_info=False):
        self.G = G  # grid dimension
        self.B = B  # buffer size
        self.Nu = Nu  # number of users
        self.num_incoming_packets_per_time_slot = num_pkts_per_tti
        self.can_users_share_position = can_users_share_position
        self.should_add_not_moving = should_add_not_moving
        self.print_debug_info = print_debug_info

        self.actions_move = one_step_moves_in_grid(
            should_add_not_moving=should_add_not_moving)

        self.indexGivenActionDictionary, self.actionGivenIndexList = createActionsDataStructures(
            self.Nu)
        self.A = len(self.actionGivenIndexList)

        self.indexGivenStateDictionary, self.stateGivenIndexList = createStatesDataStructures(
            self.actions_move, G=self.G, Nu=self.Nu, B=self.B, can_users_share_position=can_users_share_position,
            show_debug_info=self.print_debug_info)

        self.S = len(self.stateGivenIndexList)

        # self.ues_pos_prob, self.channel_spectral_efficiencies, self.ues_valid_actions = self.read_external_files()

        if True:
            # the same value for all
            self.channel_spectral_efficiencies = (
                self.num_incoming_packets_per_time_slot+1)*np.ones((G, G))
        else:
            self.channel_spectral_efficiencies = get_channel_spectral_efficiency()

        if self.print_debug_info:
            print("self.channel_spectral_efficiencies",
                  self.channel_spectral_efficiencies)

        nextStateProbability, rewardsTable = self.createEnvironment()

        # pack data structures (dic and list) to map names into indices for actions
        actions_info = [self.indexGivenActionDictionary,
                        self.actionGivenIndexList]

        # pack data structures (dic and list) to map names into indices for states
        states_info = [self.indexGivenStateDictionary,
                       self.stateGivenIndexList]

        # call superclass constructor
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable,
                                  actions_info=actions_info, states_info=states_info)
        if self.print_debug_info:
            print("INFO: finished creating environment")

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
        The action is to select one of the Nu users.
        Given that Nu=2, B=3 and G=6, there are S=xx states and A=2 actions
        The channel is fixed.
        The base station (BS) is located at the bottom-left corner.
        '''

        # now we need to populate the nextStateProbability array and rewardsTable,
        # the distribution p(s'|s,a) and expected values r(s,a,s') as in Example 3.3 of [Sutton, 2018], page 52.
        # In this case the distribution p can be stored in a 3d matrix of dimension S x A x S and the reward table in
        # another matrix with the same dimension.
        nextStateProbability = np.zeros((self.S, self.A, self.S))
        rewardsTable = np.zeros((self.S, self.A, self.S))
        # to define the mobility pattern, find all next positions
        disabled_positions = list()
        # disable the base station position: users cannot ocuppy it
        disabled_positions.append(np.array([self.G-1, 0]))
        all_valid_next_positions = all_valid_next_moves(self.G, disabled_positions,
                                                        should_add_not_moving=self.should_add_not_moving,
                                                        number_of_users=self.Nu,
                                                        can_users_share_position=self.can_users_share_position)
        for s in range(self.S):
            # current state:
            currentState = self.stateGivenIndexList[s]
            # interpret the state
            (all_positions, new_buffer_occupancy_tuple) = currentState
            # print('Reading state: positions=', all_positions,
            #      'buffers=', new_buffer_occupancy_tuple)
            for a in range(self.A):
                # currentAction = self.actionGivenIndexList[a]
                chosen_user = a  # in this case, the action is the user
                # get the channels spectral efficiency (SE)
                chosen_user_position = all_positions[chosen_user]

                se = self.channel_spectral_efficiencies[chosen_user_position[0],
                                                        chosen_user_position[1]]
                # based on selected (chosen) user, update its buffer
                num_packets_supported_by_channel = se  # it is not the transmitted packets

                # Update buffer according to action and packet transmission
                new_buffer = np.array(new_buffer_occupancy_tuple)
                # decrement buffer of chosen user
                new_buffer[chosen_user] -= num_packets_supported_by_channel

                # in case there were less packets in buffer than the channel supports
                if new_buffer[chosen_user] < 0:
                    num_unavailable_packets = -new_buffer[chosen_user]
                    num_transmitted_packets = num_packets_supported_by_channel - num_unavailable_packets
                    new_buffer[chosen_user] = 0
                else:
                    num_transmitted_packets = num_packets_supported_by_channel

                # Update buffer based on arrival of new packets
                new_buffer += self.num_incoming_packets_per_time_slot  # arrival of new packets

                # check if buffer overflow occurred
                # in case positive, limit the buffers to maximum capacity
                number_dropped_packets = new_buffer - self.B
                # no packets dropped in this case
                number_dropped_packets[number_dropped_packets < 0] = 0

                # take in account buffer capacity and update in case packets were dropped
                new_buffer -= number_dropped_packets

                # convert to tuple to compose part of the new state
                new_buffer_occupancy_tuple = tuple(new_buffer)

                # calculate reward
                sumDrops = np.sum(number_dropped_packets)
                r = 10-sumDrops  # option 1
                # r = num_transmitted_packets-sumDrops  # option 2

                # probabilistic part: consider the user mobility
                valid_next_positions = all_valid_next_positions[all_positions]
                # define a probability value to each new position
                num_valid_next_positions = len(valid_next_positions)
                # impose uniform probability
                prob = 1.0 / num_valid_next_positions
                for next_pos in valid_next_positions:
                    # compose the state
                    new_state = (next_pos, new_buffer_occupancy_tuple)
                    nextStateIndice = self.indexGivenStateDictionary[new_state]

                    nextStateProbability[s, a, nextStateIndice] = prob
                    rewardsTable[s, a, nextStateIndice] = r

        return nextStateProbability, rewardsTable

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
        if printPostProcessingInfo:  # use self.print_debug_info
            print(self.bitRates, self.packetDropCounts)
        # print('accumulated rates =', self.bitRates, ' drops =', self.packetDropCounts)
        # print('kkkk = ', s, action, reward, nexts)

    def resetCounters(self):
        # reset counters
        self.bitRates = np.zeros((self.M,))
        self.packetDropCounts = np.zeros((self.M,))


def createActionsDataStructures(Nu) -> tuple[dict, list]:
    """
    Nu is the number of users.
    Example assuming Nu=2 users: ['u0', 'u1']
    """
    possibleActions = list()
    for u in range(Nu):
        possibleActions.append("u" + str(u))
    dictionaryGetIndex = dict()
    listGivenIndex = list()
    for uniqueIndex in range(len(possibleActions)):
        dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
        listGivenIndex.append(possibleActions[uniqueIndex])
    return dictionaryGetIndex, listGivenIndex


def init_next_state_probability():
    pass


def init_rewards():
    pass


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


def createStatesDataStructures(possible_movements, G=6, Nu=2, B=3, can_users_share_position=False, show_debug_info=True) -> tuple[dict, list]:
    # G is the axis dimension, on both horizontal and vertical
    # I cannot use below:
    # all_positions_list = list(itertools.product(np.arange(G), repeat=2))
    # because the base station is at position (G-1, 0) and users cannot be
    # in the same position at the same time
    if show_debug_info:
        num_states = (B+1)**Nu  # initialize
        if can_users_share_position:
            for i in range(Nu):
                num_states *= (G**2)-(i+1)
        else:
            num_states *= (G**2-1)**Nu
        print("theoretical number of states =", num_states)

    bs_position = (G-1, 0)  # Base station position
    disabled_positions = list()
    disabled_positions.append(bs_position)

    all_positions_list = combined_users_positions(G,
                                                  disabled_positions,
                                                  possible_movements,
                                                  number_of_users=Nu,
                                                  can_users_share_position=can_users_share_position)

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
    print("Running main of simple_user_scheduling_env.py")
    print("Creating the UserSchedulingEnv environment... It takes some time.")

    use_saved_file = False
    file_path = "UserSchedulingEnv.pickle"

    if use_saved_file and os.path.exists(file_path):
        print(f"The file {file_path} exists. I will read it")
        env = pickle.load(open(file_path, "rb"))
    else:
        if use_saved_file:
            print(
                f"The file {file_path} does not exist. I will create the object and create the file")
        env = UserSchedulingEnv(
            G=3, B=2, Nu=2, num_pkts_per_tti=1,
            can_users_share_position=False,
            should_add_not_moving=False,
            print_debug_info=True)  # G is the grid size
        pickle.dump(env, open(file_path, "wb"))

    # env.prettyPrint()
    # check_matrix(env.nextStateProbability) # it was ok

    print("Example of action:", env.actionListGivenIndex[0])
    print("Example of state:", env.stateListGivenIndex[30])

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
