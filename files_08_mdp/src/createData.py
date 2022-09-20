import numpy as np
import itertools

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

        N = len(all_states) #number of states
        stateGivenIndexList = list()
        indexGivenStateDictionary = dict()
        uniqueIndex = 0
        # add states to both dictionary and its inverse mapping (a list)
        for i in range(N):
            stateGivenIndexList.append(all_states[i])
            indexGivenStateDictionary[all_states[i]] = uniqueIndex
            uniqueIndex += 1

        return indexGivenStateDictionary, stateGivenIndexList