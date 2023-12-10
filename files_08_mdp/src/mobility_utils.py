import numpy as np
import itertools


def combined_users_positions(grid_dimension: int,
                             disabled_positions: list,
                             possible_movements: list,
                             number_of_users=1,
                             can_users_share_position=False) -> list:
    '''
    Based on a list of possible_movements for a single user,
    builds a list of combined positions for the given number of users.
    For instance, the returned list can have an entry:
    ((1, 2), (1, 0), (2, 2), (0, 1))
    when the number_of_users is 4. Note that no position is shared by
    users because can_users_share_position=False in this case.
    Another example, with number_of_users=2 and can_users_share_position=True is:
    ((3, 2), (3, 2))
    '''
    next_movements = single_user_moves(grid_dimension,
                                       disabled_positions,
                                       possible_movements)
    if number_of_users == 1:  # no need to do anything else
        return next_movements
    all_single_user_positions = next_movements.keys()

    combined_positions = list(itertools.product(
        all_single_user_positions, repeat=number_of_users))
    if can_users_share_position:
        valid_combined_positions = combined_positions
    else:
        # need to eliminate cases in which more than one user share the same position
        valid_combined_positions = list()
        for positions in combined_positions:
            # check if positions does not have repeated elements
            # hint: https://www.trainingint.com/how-to-find-duplicates-in-a-python-list.html
            if len(set(positions)) == len(positions):
                valid_combined_positions.append(positions)

    return valid_combined_positions


def combined_users_next_positions(valid_combined_positions: list, next_movements: dict):
    # iterate over all valid combination of users' positions
    num_users = len(valid_combined_positions[0])
    valid_next_positions_dict = dict()
    for positions in valid_combined_positions:
        all_next_positions = list()  # list all possible next positions
        for u in range(num_users):  # add next positions of each user to list
            # position for user u, example (1,2)
            user_current_position = positions[u]
            user_next_positions = next_movements[user_current_position]
            user_next_positions = flatten_nested_lists(user_next_positions)
            all_next_positions.append(user_next_positions)
        # combined_next_positions = flatten_nested_lists(
        #    cartesian_product(all_next_positions))
        combined_next_positions = cartesian_product(all_next_positions)
        # print(positions, "=>", combined_next_positions)

        # check whether or not all combinations are valid
        valid_next_positions = list()
        for next_positions in combined_next_positions:
            next_positions = tuple(next_positions)
            if next_positions in valid_combined_positions:
                valid_next_positions.append(next_positions)

        # add to the dict
        valid_next_positions_dict[positions] = valid_next_positions
    return valid_next_positions_dict


def single_user_moves(grid_dimension: int,
                      disabled_positions: list,
                      possible_movements: list) -> dict:
    '''
    Assume a single user.
    Assume world is a grid corresponding to a matrix of size grid_dimension x grid_dimension.
    Assume top-left corner is [0, 0].
    '''
    if disabled_positions == None:
        D = 0
    else:
        D = len(disabled_positions)

    next_movements = dict()
    for i in range(grid_dimension):
        for j in range(grid_dimension):
            current_position = np.array([i, j])
            position_is_invalid = False
            for d in range(D):
                if np.array_equal(current_position, disabled_positions[d]):
                    position_is_invalid = True
                    break
            if position_is_invalid:
                continue  # go to next position
            # use tuple, do not use np.array([i, j]) because ndarray is not hashable
            current_position = tuple(current_position)
            next_movements[current_position] = list()
            valid_next_positions = moves_from_position(
                grid_dimension, current_position, disabled_positions, possible_movements)
            next_movements[current_position].append(valid_next_positions)
    return next_movements


def flatten_nested_lists(lst):
    '''
    I get [[[array([1, 2])], [array([1, 0])]]] but would like to have [array([1, 2]), array([1, 0])].
    From ChatGPT
    '''
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_nested_lists(item))
        else:
            result.append(item)
    return result


def cartesian_product(lists):
    '''
    Cartesian product of list with nested lists.
    Example of input: 
    lists = [ [1, 2],
            [3, 4, 5],
            [6] ]
    print(cartesian_product(lists))
    Output:
            [[1, 3, 6], [2, 3, 6], [1, 4, 6], [2, 4, 6], [1, 5, 6], [2, 5, 6]]
    '''
    num_users = len(lists)
    # find dimension of output array
    dimensions = np.zeros((num_users,), dtype=int)
    num_combinations = 1
    for u in range(num_users):
        this_list = lists[u]
        dimensions[u] = len(this_list)
        num_combinations *= dimensions[u]
    # generate all combinations
    # combinations = np.zeros((num_combinations, num_users), dtype=int)
    combinations = list()
    for i in range(num_combinations):
        combinations.append(list())
        for _ in range(num_users):
            combinations[i].append(None)

    for u in range(num_users):
        combination_counter = 0
        num_repetitions = num_combinations // np.prod(dimensions[u:])
        this_list = lists[u]
        while combination_counter < num_combinations:
            for item in this_list:
                for _ in range(num_repetitions):
                    combinations[combination_counter][u] = item
                    combination_counter += 1
    return combinations


def moves_from_position(grid_dimension: int, current_position: np.ndarray,
                        disabled_positions: list, possible_movements: list) -> list:
    '''
    Assumes a single user.
    This only imposes the mobility pattern.
    Do not talk about actions and states, but positions.
    Assume top-left corner is [0, 0].
    '''
    M = len(possible_movements)
    if disabled_positions == None:
        D = 0
    else:
        D = len(disabled_positions)

    i = current_position[0]  # row
    j = current_position[1]  # column
    if i < 0 or j < 0 or i > grid_dimension-1 or j > grid_dimension-1:
        raise Exception("Position " + str(current_position) +
                        " is invalid if grid dimension is " + str(grid_dimension))

    valid_next_positions = list()
    for m in range(M):
        movement = possible_movements[m]
        new_position = current_position + movement
        i = new_position[0]  # row
        j = new_position[1]  # column
        if i < 0 or j < 0 or i > grid_dimension-1 or j > grid_dimension-1:
            continue  # the new position is not valid
        # now we check if new position was disabled by function caller
        new_position_is_valid = True
        for d in range(D):
            if np.array_equal(new_position, disabled_positions[d]):
                new_position_is_valid = False
                break
        if new_position_is_valid:
            valid_next_positions.append(tuple(new_position))

    return valid_next_positions


def one_step_moves_in_grid(should_add_not_moving=False) -> list:
    # define possible movements
    possible_movements = list()
    possible_movements.append(np.array([1, 0]))  # go down
    possible_movements.append(np.array([0, 1]))  # go right
    possible_movements.append(np.array([0, -1]))  # go left
    possible_movements.append(np.array([-1, 0]))  # go up
    if should_add_not_moving:
        possible_movements.append(np.array([0, 0]))  # stay still, do not move
    return possible_movements


def all_valid_next_moves(grid_dimension: int, disabled_positions: list,
                         should_add_not_moving=True,
                         number_of_users=2,
                         can_users_share_position=False) -> tuple[list, list]:
    # define possible movements
    possible_movements = one_step_moves_in_grid(
        should_add_not_moving=should_add_not_moving)

    valid_combined_positions = combined_users_positions(grid_dimension,
                                                        disabled_positions,
                                                        possible_movements,
                                                        number_of_users=number_of_users,
                                                        can_users_share_position=can_users_share_position)

    next_movements = single_user_moves(grid_dimension,
                                       disabled_positions,
                                       possible_movements)

    valid_next_positions = combined_users_next_positions(
        valid_combined_positions, next_movements)

    # for positions, nextpositions in valid_next_positions.items():
    #    print(positions, "=>", nextpositions)
    return valid_next_positions


if __name__ == '__main__':
    grid_dimension = 3
    current_position = np.array([0, 1])
    disabled_positions = list()
    disabled_positions.append(np.array([0, 0]))
    disabled_positions.append(np.array([1, 1]))
    disabled_positions.append(np.array([2, 1]))
    # disabled_positions = None

    possible_movements = one_step_moves_in_grid(
        should_add_not_moving=False)
    valid_combined_positions = combined_users_positions(grid_dimension,
                                                        disabled_positions,
                                                        possible_movements,
                                                        number_of_users=2,
                                                        can_users_share_position=False)
    print("valid_combined_positions=", valid_combined_positions)
    print("len(valid_combined_positions)=", len(valid_combined_positions))

    next_movements = single_user_moves(grid_dimension,
                                       disabled_positions,
                                       possible_movements)
    print("next_movements=", next_movements)

    valid_next_positions = combined_users_next_positions(
        valid_combined_positions, next_movements)
    print("valid_next_positions=", valid_next_positions)

    all_valid_next_moves(grid_dimension, disabled_positions,
                         should_add_not_moving=True,
                         number_of_users=2,
                         can_users_share_position=False)
