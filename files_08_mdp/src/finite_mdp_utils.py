'''
This class implements methods to deal with 
a finite Markov Decision Process (MDP) for tabular RL.
Look at the definition of a finite MDP in page 49 of
Sutton & Barto's book, version 2018, with 550 pages.

    Note that a policy is represented here as a distribution over the
    possible actions for each state and stored as an array of dimension S x A.
    A general policy may be represented in other ways. But considering the
    adopted representation here, the "policy" coincides with the distribution
    of actions for each state.

The methods are "static", in the sense that are not defined within a class.

Note that a policy is represented here as a matrix S x A, providing a distribution
over the possible actions for each state. A matrix with the state values can be
easily converted into a policy. 

Aldebaro. Oct 25, 2023.
'''
from __future__ import print_function
import numpy as np
# from builtins import print
from random import choices
import gymnasium as gym
from gymnasium import spaces

from known_dynamics_env import KnownDynamicsEnv, SimpleKnownDynamicsEnv, RandomKnownDynamicsEnv, RecycleRobotEnv


def check_if_fmdp(environment: gym.Env):
    # checks if env is a FMDP gym with discrete spaces
    assert isinstance(environment.action_space, spaces.Discrete)
    assert isinstance(environment.observation_space, spaces.Discrete)


def get_space_dimensions_for_openai_gym(environment: gym.Env) -> tuple[int, int]:
    '''
    Returns S and A.
    '''
    assert isinstance(environment.action_space, spaces.Discrete)
    assert isinstance(environment.observation_space, spaces.Discrete)
    S = environment.observation_space.n
    A = environment.action_space.n
    return S, A


def compute_state_values(env: gym.Env, policy: np.ndarray, in_place=False,
                         discountGamma=0.9) -> tuple[np.ndarray, int]:
    '''
    Iterative policy evaluation. Page 75 of [Sutton, 2018].
    Here a policy (not necessarily optimum) is provided.
    It can generate, for instance, Fig. 3.2 in [Sutton, 2018]
    '''
    S = env.S
    A = env.A
    # (S, A, nS) = self.environment.nextStateProbability.shape
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
                    p = env.nextStateProbability[s, a, nexts]
                    r = env.rewardsTable[s, a, nexts]
                    value += policy[s, a] * p * \
                        (r + discountGamma * src[nexts])
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


def compute_optimal_state_values_nonsparse(env: KnownDynamicsEnv, discountGamma=0.9) -> tuple[np.ndarray, int]:
    '''Page 63 of [Sutton, 2018], Eq. (3.19)'''
    # assert isinstance(env, KnownDynamicsEnv)
    S = env.S
    A = env.A
    # (S, A, nS) = env.nextStateProbability.shape
    # A = len(actionListGivenIndex)
    new_state_values = np.zeros((S,))
    state_values = np.zeros((S,))
    iteration = 1
    a_candidates = np.zeros((A,))
    while True:
        for s in range(S):
            # fill with zeros without creating new array
            # a_candidates[:] = 0
            a_candidates.fill(0.0)
            for a in range(A):
                value = 0
                for nexts in range(S):
                    p = env.nextStateProbability[s, a, nexts]
                    if p != 0:
                        r = env.rewardsTable[s, a, nexts]
                        value += p * (r + discountGamma * state_values[nexts])
                a_candidates[a] = value
            new_state_values[s] = np.max(a_candidates)
        improvement = np.sum(np.abs(new_state_values - state_values))
        # print('improvement =', improvement)
        if False:  # debug
            print('state values=', state_values)
            print('new state values=', new_state_values)
            print('it=', iteration, 'improvement = ', improvement)
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            state_values[i] = new_state_values[i]
        if improvement < 1e-4:
            break

        iteration += 1

    return state_values, iteration


def compute_optimal_state_values(env: KnownDynamicsEnv, discountGamma=0.9, use_nonsparse_version=False) -> tuple[np.ndarray, int]:
    '''
    This is useful when nextStateProbability is sparse. It only goes over the
    next states that are feasible.
    Page 63 of [Sutton, 2018], Eq. (3.19)'''
    if use_nonsparse_version:
        return compute_optimal_state_values_nonsparse(env, discountGamma)

    assert isinstance(env, KnownDynamicsEnv)

    S = env.S

    # (S, A, nS) = env.nextStateProbability.shape
    # A = len(actionListGivenIndex)
    new_state_values = np.zeros((S,))
    state_values = np.zeros((S,))
    iteration = 1
    valid_next_states = env.valid_next_states
    while True:
        for s in range(S):

            possibleAction = env.possible_actions_per_state[s] #Getting the Possible actions per state
            a_candidates = np.zeros(len(possibleAction)) #Creating a array to compute the possibles actions 
            # print(s)
            # fill with zeros without creating new array
            # a_candidates[:] = 0
            a_candidates.fill(0.0)
            feasible_next_states = valid_next_states[s]
            num_of_feasible_next_states = len(feasible_next_states)
            for a in possibleAction:
                value = 0
                # for nexts in range(S):
                for feasible_nexts in range(num_of_feasible_next_states):
                    # print('feasible_nexts=',feasible_nexts)
                    nexts = feasible_next_states[feasible_nexts]
                    p = env.nextStateProbability[s, a, nexts]
                    r = env.rewardsTable[s, a, nexts]
                    # print('p',p,'state_values[nexts]',state_values[nexts],'r',r)
                    value += p * (r + discountGamma * state_values[nexts])
                a_candidates[a] = value
            new_state_values[s] = np.max(a_candidates)
        improvement = np.sum(np.abs(new_state_values - state_values))
        # print('improvement =', improvement)
        if True:  # debug AK
            print('state values=', state_values)
            print('new state values=', new_state_values)
            print('it=', iteration, 'improvement = ', improvement)
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            state_values[i] = new_state_values[i]
        if improvement < 1e-4:
            break

        iteration += 1

    return state_values, iteration


def compute_optimal_action_values_nonsparse(env: KnownDynamicsEnv,
                                            discountGamma=0.9,
                                            tolerance=1e-20) -> tuple[np.ndarray, np.ndarray]:
    '''
    In [Sutton, 2018] the main result of this method in called "the optimal
    action-value function" and defined in Eq. (3.16) in page 63.
    Page 64 of [Sutton, 2018], Eq. (3.20)'''

    assert isinstance(env, KnownDynamicsEnv)
    S = env.S
    A = env.A
    # (S, A, nS) = env.nextStateProbability.shape
    # A = len(actionListGivenIndex)
    new_action_values = np.zeros((S, A))
    action_values = new_action_values.copy()
    iteration = 1
    stopping_criteria = list()
    while True:
        # src = new_action_values if in_place else action_values
        for s in range(S):
            for a in range(A):
                value = 0
                for nexts in range(S):
                    # p(s'|s,a) = p(nexts|s,a)
                    p = env.nextStateProbability[s, a, nexts]
                    # r(s,a,s') = r(s,a,nexts)
                    r = env.rewardsTable[s, a, nexts]
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
        stopping_criteria.append(improvement)
        if improvement <= tolerance:
            action_values = new_action_values.copy()
            break

        action_values = new_action_values.copy()
        iteration += 1

    return action_values, np.array(stopping_criteria)


def compute_optimal_action_values(env: KnownDynamicsEnv,
                                  discountGamma=0.9,
                                  use_nonsparse_version=False,
                                  tolerance=1e-20) -> tuple[np.ndarray, np.ndarray]:
    '''
    In [Sutton, 2018] the main result of this method in called "the optimal
    action-value function" and defined in Eq. (3.16) in page 63.

    Assumes sparsity.
    Page 64 of [Sutton, 2018], Eq. (3.20)'''
    if use_nonsparse_version:
        return compute_optimal_action_values_nonsparse(env, discountGamma=discountGamma, tolerance=tolerance)

    assert isinstance(env, KnownDynamicsEnv)
    S = env.S
    A = env.A
    new_action_values = np.zeros((S, A))
    action_values = np.zeros((S, A))
    iteration = 1
    valid_next_states = env.valid_next_states
    stopping_criteria_per_iteration = list()
    while True:
        for s in range(S):
            feasible_next_states = valid_next_states[s]
            num_of_feasible_next_states = len(feasible_next_states)
            for a in range(A):
                value = 0
                for feasible_nexts in range(num_of_feasible_next_states):
                    nexts = feasible_next_states[feasible_nexts]
                    # p(s'|s,a) = p(nexts|s,a)
                    p = env.nextStateProbability[s, a, nexts]
                    # r(s,a,s') = r(s,a,nexts)
                    r = env.rewardsTable[s, a, nexts]
                    best_a = np.max(action_values[nexts])
                    value += p * (r + discountGamma * best_a)
                new_action_values[s, a] = value

        # check if should stop because process converged
        abs_difference_array = np.abs(new_action_values - action_values)
        stopping_criterion_option = 2  # 1 uses sum and 2 uses max
        if stopping_criterion_option == 1:
            # use the sum
            stopping_criterion = np.sum(
                abs_difference_array)/np.max(np.abs(new_action_values))
        else:
            # use the max
            stopping_criterion = np.max(
                abs_difference_array)/np.max(np.abs(new_action_values))
        stopping_criteria_per_iteration.append(stopping_criterion)
        # print('DEBUG: stopping_criterion =', stopping_criterion)
        if False:  # debug
            print('state values=', action_values)
            print('new state values=', new_action_values)
            print('it=', iteration, 'improvement = ', stopping_criterion)

        # action_values = new_action_values.copy()
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            for j in range(A):
                action_values[i, j] = new_action_values[i, j]

        if stopping_criterion <= tolerance:
            # print('DEBUG: stopping_criterion =',
            #      stopping_criterion, "and tolerance=", tolerance)
            break

        iteration += 1

    return action_values, np.array(stopping_criteria_per_iteration)


def convert_action_values_into_policy(action_values: np.ndarray) -> np.ndarray:
    (S, A) = action_values.shape
    policy = np.zeros((S, A))
    for s in range(S):
        maxPerState = max(action_values[s])
        maxIndices = np.where(action_values[s] == maxPerState)
        # maxIndices is a tuple and we want to get first element maxIndices[0]
        # impose uniform distribution
        policy[s, maxIndices] = 1.0 / len(maxIndices[0])
    return policy


def get_uniform_policy_for_fully_connected(S: int, A: int) -> np.ndarray:
    '''
    Assumes all actions can be performed at each state.
    See @get_uniform_policy_for_known_dynamics for
    an alternative that takes in account the dynamics
    of the defined environment.
    '''
    policy = np.zeros((S, A))
    uniformProbability = 1.0 / A
    for s in range(S):
        for a in range(A):
            policy[s, a] = uniformProbability
    return policy


def get_uniform_policy_for_known_dynamics(env: KnownDynamicsEnv) -> np.ndarray:
    '''
    Takes in account the dynamics of the defined environment
    when defining actions that can be performed at each state.
    See @get_uniform_policy_for_fully_connected for
    an alternative that does not have restriction.
    '''
    assert isinstance(env, KnownDynamicsEnv)
    policy = np.zeros((env.S, env.A))
    for s in range(env.S):
        # possible_actions_per_state is a list of lists, that indicates for each state, the list of allowed actions
        valid_actions = env.possible_actions_per_state[s]
        # no problem if denominator is zero
        uniform_probability = 1.0 / len(valid_actions)
        for a in range(len(valid_actions)):
            policy[s, a] = uniform_probability
    return policy


def run_several_episodes(env: gym.Env, policy: np.ndarray, num_episodes=10, max_num_time_steps_per_episode=100, printInfo=False, printPostProcessingInfo=False) -> np.ndarray:
    '''
    Runs num_episodes episodes and returns their rewards.'''
    rewards = np.zeros(num_episodes)
    for e in range(num_episodes):
        rewards[e] = run_episode(env, policy, maxNumIterations=max_num_time_steps_per_episode,
                                 printInfo=printInfo, printPostProcessingInfo=printPostProcessingInfo)
    return rewards


def run_episode(env: gym.Env, policy: np.ndarray, maxNumIterations=100, printInfo=False, printPostProcessingInfo=False) -> int:
    '''
    Reset and runs a complete episode for the environment env according to
    the specified policy.
    The policy already has distribution probabilities that specify the
    valid actions, so we do not worry about it (we do not need to invoke
    methods such as @choose_epsilon_greedy_action)
    '''
    env.reset()
    s = env.get_state()
    totalReward = 0
    list_of_actions = np.arange(env.A)
    if printInfo:
        print('Initial state = ', env.stateListGivenIndex[s])
    for it in range(maxNumIterations):
        policy_weights = np.squeeze(policy[s])
        sumWeights = np.sum(policy_weights)
        if sumWeights == 0:
            print(
                "Warning: reached state that does not have a valid action to take. Ending the episode")
            break
        action = choices(list_of_actions, weights=policy_weights, k=1)[0]
        ob, reward, gameOver, history = env.step(action)

        # assume the user may want to apply some postprocessing step, similar to a callback function
        env.postprocessing_MDP_step(history, printPostProcessingInfo)
        if printInfo:
            print(history)
        totalReward += reward
        s = env.get_state()  # update current state
        if gameOver == True:
            break
    if printInfo:
        print('totalReward = ', totalReward)
    return totalReward


def get_unrestricted_possible_actions_per_state(env: gym.Env) -> list:
    '''
    Create a list of lists, that indicates for each state, the list of allowed actions.
    Here we indicate all actions are valid for each state.
    '''
    S, A = get_space_dimensions_for_openai_gym(env)
    possible_actions_per_state = list()
    for s in range(S):
        possible_actions_per_state.append(list())
        for a in range(A):
            possible_actions_per_state[s].append(a)
    return possible_actions_per_state


def q_learning_episode(env: gym.Env,
                       stateActionValues: np.ndarray,
                       possible_actions_per_state: list,
                       max_num_time_steps=100, stepSizeAlpha=0.1,
                       explorationProbEpsilon=0.01, discountGamma=0.9) -> int:
    '''    
    An episode with Q-Learning. We reset the environment.
    Note stateActionValues is not reset within this method, and can be already initialized.
    One needs to pay attention to allowing only
    valid actions. The simple algorithms provided in Sutton's book do not
    take that in account. See the discussion:
    https://ai.stackexchange.com/questions/31819/how-to-handle-invalid-actions-for-next-state-in-q-learning-loss
    Here we use possible_actions_per_state to indicate the valid actions.
    This list is a member variable of KnownDynamicsEnv enviroments but must be generated
    for other enviroments with the method @get_unrestricted_possible_actions_per_state
    or a similar one, which takes in account eventual restrictions.
    @return: average reward within this episode
    '''
    env.reset()
    currentState = env.get_state()
    rewards = 0.0
    for numIterations in range(max_num_time_steps):
        currentAction = action_via_epsilon_greedy(currentState, stateActionValues,
                                                  possible_actions_per_state,
                                                  explorationProbEpsilon=explorationProbEpsilon,
                                                  run_faster=False)

        newState, reward, gameOver, history = env.step(currentAction)
        rewards += reward
        # Q-Learning update
        stateActionValues[currentState, currentAction] += stepSizeAlpha * (
            reward + discountGamma * np.max(stateActionValues[newState, :]) -
            stateActionValues[currentState, currentAction])
        currentState = newState
        if gameOver:
            break
    # normalize rewards to facilitate comparison
    return rewards / (numIterations+1)


def action_via_epsilon_greedy(state: int,
                              stateActionValues: np.ndarray,
                              possible_actions_per_state: list,
                              explorationProbEpsilon=0.01, run_faster=False) -> int:
    '''
    Choose an action based on epsilon greedy algorithm.
    '''
    if np.random.binomial(1, explorationProbEpsilon) == 1:
        # explore among valid options
        return np.random.choice(possible_actions_per_state[state])
    else:
        # exploit, choosing an action with maximum value
        return action_greedy(state, stateActionValues, possible_actions_per_state, run_faster=run_faster)


def action_greedy(state: int,
                  stateActionValues: np.ndarray,
                  possible_actions_per_state: list,
                  run_faster=False) -> int:
    '''
    Greedly choose an action with maximum value.
    '''
    values_for_given_state = stateActionValues[state]
    if run_faster == True:
        # always return the first index with maximum value
        # this may create a problem for the agent to explore other options
        # or choose an invalid option
        max_index = np.argmax(values_for_given_state)
    else:
        actions_for_given_state = possible_actions_per_state[state]
        # make sure the action is valid, but keeping invalid actions with value=-infinity
        valid_values_for_given_state = -np.Inf * \
            np.ones(values_for_given_state.shape)
        valid_values_for_given_state[actions_for_given_state] = values_for_given_state[actions_for_given_state]
        max_value = np.max(valid_values_for_given_state)
        # Use numpy.where to get all indices where the array is equal to its maximum value
        all_max_indices = np.where(
            valid_values_for_given_state == max_value)[0]
        max_index = np.random.choice(all_max_indices)
    return max_index


def q_learning_several_episodes(env: gym.Env,
                                episodes_per_run=100,
                                max_num_time_steps_per_episode=10,
                                num_runs=1,
                                stepSizeAlpha=0.1, explorationProbEpsilon=0.01,
                                possible_actions_per_state=None, verbosity=1) -> tuple[np.ndarray, np.ndarray]:
    '''Use independent runs instead of a single run.
    Increase num_runs if you want smooth numbers representing the average.
    @return tuple with stateActionValues corresponding to best average rewards among all runs
    and '''
    if verbosity > 0:
        print("Running", num_runs, " runs of q_learning_several_episodes() with",
              episodes_per_run, "episodes per run")

    if possible_actions_per_state == None:
        if isinstance(env, KnownDynamicsEnv):
            possible_actions_per_state = env.possible_actions_per_state
        else:
            # assume it is a generic OpenAI gym and all actions are possible for all states
            # if this is not the case, the user should build possible_actions_per_state himself/herself
            # and pass to this method
            possible_actions_per_state = get_unrestricted_possible_actions_per_state(
                env)

    # shows convergence over episodes
    rewardsQLearning = np.zeros(episodes_per_run)
    best_stateActionValues = np.zeros((env.S, env.A))  # store best over runs
    largest_reward = -np.Inf  # initialize with negative value
    for run in range(num_runs):
        stateActionValues = np.zeros((env.S, env.A))  # reset for each run
        sum_rewards_this_run = 0
        for i in range(episodes_per_run):
            # update stateActionValues in-place (that is, updates stateActionValues)
            reward = q_learning_episode(env, stateActionValues,
                                        possible_actions_per_state,
                                        max_num_time_steps=max_num_time_steps_per_episode,
                                        stepSizeAlpha=stepSizeAlpha,
                                        explorationProbEpsilon=explorationProbEpsilon)
            rewardsQLearning[i] += reward
            sum_rewards_this_run += reward
        average_reward_this_run = sum_rewards_this_run / episodes_per_run
        if average_reward_this_run > largest_reward:
            largest_reward = average_reward_this_run
            best_stateActionValues = stateActionValues.copy()
            if verbosity > 0:
                print("Found better stateActionValues")
        if verbosity > 0:
            print('run=', run, 'average reward=', average_reward_this_run)
    # need to normalize to get rewards convergence over episodes
    rewardsQLearning /= num_runs
    if verbosity > 1:
        print('rewardsQLearning = ', rewardsQLearning)
        print('newStateActionValues = ', stateActionValues)
        print('qlearning_policy = ',
              convert_action_values_into_policy(stateActionValues))
    return best_stateActionValues, rewardsQLearning


def create_random_next_state_probability(S: int, A: int) -> np.ndarray:
    nextStateProbability = np.random.rand(S, A, S)  # positive numbers
    for s in range(S):
        for a in range(A):
            pmf = nextStateProbability[s, a]  # probability mass function (pmf)
            total_prob = sum(pmf)
            if total_prob == 0:
                # arbitrarily, make first state have all probability
                nextStateProbability[s, a, 0] = 1
            else:
                # normalize to have a pmf
                nextStateProbability[s, a] /= total_prob
    return nextStateProbability


def generate_trajectory(env: gym.Env, policy: np.ndarray, num_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    We generate trajectories according to the specified policy.
    Note that actions are generated by the agent, not by the environment. 
    At time t=0 one obtains the reward for t=1 (convention of Sutton's book).
    We do not reset the environment env. It starts from its current state.
    The policy already has distribution probabilities that specify the
    valid actions, so we do not worry about it (we do not need to invoke
    methods such as @choose_epsilon_greedy_action)
    '''
    list_of_actions = np.arange(env.A)  # list all existing actions
    # initialize arrays
    taken_actions = np.zeros(num_steps, dtype=int)
    rewards_tp1 = np.zeros(num_steps)
    states = np.zeros(num_steps, dtype=int)
    for t in range(num_steps):
        states[t] = env.current_observation_or_state  # current state
        # choose action according to the policy
        taken_actions[t] = choices(
            list_of_actions, weights=policy[states[t]])[0]
        ob, reward, gameOver, history = env.step(taken_actions[t])
        # at time t=0 one obtains the reward for t=1 (convention of Sutton's book)
        rewards_tp1[t] = reward
    return taken_actions, rewards_tp1, states


def format_trajectory_as_single_array(taken_actions: np.ndarray, rewards_tp1: np.ndarray, states: np.ndarray) -> np.ndarray:
    '''
    Format as a single vector
    '''
    T = len(taken_actions)
    # pre-allocate space for S0, A0, R1, S1, A1, R2, S2, A2, R3, ...
    trajectory = np.zeros(3*T)
    for t in range(T):
        trajectory[3*t] = states[t]
        trajectory[3*t+1] = taken_actions[t]
        trajectory[3*t+2] = rewards_tp1[t]
    return trajectory


def print_trajectory(trajectory: np.ndarray):
    '''
    Indices must allow to interpret the trajectory according
    to convention in Sutton & Barto's book, where after the
    action at time t, we obtain the reward of time t+1.'''
    T = len(trajectory) // 3
    t = 0
    print("time=" + str(t) + "  R in undefined" + "  S" + str(t) + "=" +
          str(int(trajectory[3*t])) + "  A" + str(t) + "=" + str(int(trajectory[3*t+1])))
    for t in range(1, T):
        print("time=" + str(t) +
              "  R" + str(t) + "=" + str(trajectory[3*(t-1)+2]) +
              "  S" + str(t) + "=" + str(int(trajectory[3*t])) + "  A" + str(
            t) + "=" + str(int(trajectory[3*t+1])))
    print("time=" + str(T) + "  Last reward R" +
          str(T) + "=" + str(trajectory[-1]))


def pretty_print_policy(env: KnownDynamicsEnv, policy: np.ndarray):
    '''
    Print policy.
    '''
    for s in range(env.S):
        currentState = env.stateListGivenIndex[s]
        print('\ns' + str(s) + '=' + str(currentState))
        first_action = True
        for a in range(env.A):
            if policy[s, a] == 0:
                continue
            currentAction = env.actionListGivenIndex[a]
            if first_action:
                print(' | a' + str(a) + '=' + str(currentAction), end='')
                first_action = False  # disable this way of printing
            else:
                print(' or a' + str(a) + '=' + str(currentAction), end='')
    print("")


def hyperparameter_grid_search(env: gym.Env):
    '''
    Grid search over alphas and epsilons
    '''
    alphas = (0.1, 0.5, 0.99)
    epsilons = (0.1, 0.001)
    print("Started grid search over alphas and epsilons")
    for a in alphas:
        for e in epsilons:
            env.reset()
            action_values, rewardsQLearning = q_learning_several_episodes(
                env, num_runs=1, episodes_per_run=2, stepSizeAlpha=a, explorationProbEpsilon=e,
                max_num_time_steps_per_episode=5, verbosity=0)
            print("Reward =", np.mean(rewardsQLearning),
                  " for grid search for alpha=", a, 'epsilon=', e)
            # fileName = 'smooth_q_eps' + str(e) + '_alpha' + str(a) + '.txt'
            # sys.stdout = open(fileName, 'w')


def compare_q_learning_with_optimum_policy(env: KnownDynamicsEnv,
                                           max_num_time_steps_per_episode=100,
                                           num_episodes=10,
                                           explorationProbEpsilon=0.2,
                                           output_files_prefix=None):
    # find and use optimum policy
    env.reset()
    tolerance = 1e-10
    action_values, stopping_criteria = compute_optimal_action_values(
        env, tolerance=tolerance)
    iteration = stopping_criteria.shape[0]
    stopping_criterion = stopping_criteria[-1]
    print("\nMethod compute_optimal_action_values() converged in",
          iteration, "iterations with stopping criterion=", stopping_criterion)
    optimum_policy = convert_action_values_into_policy(action_values)
    optimal_rewards = run_several_episodes(env, optimum_policy,
                                           max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                           num_episodes=num_episodes)
    average_reward = np.mean(optimal_rewards)
    stddev_reward = np.std(optimal_rewards)
    print('\nUsing optimum policy, average reward=',
          average_reward, ' standard deviation=', stddev_reward)

    # learn a policy with Q-learning. Use a single run.
    stateActionValues, rewardsQLearning = q_learning_several_episodes(
        env, num_runs=1, episodes_per_run=num_episodes,
        max_num_time_steps_per_episode=max_num_time_steps_per_episode,
        explorationProbEpsilon=explorationProbEpsilon)
    print('stateActionValues:', stateActionValues)
    print('rewardsQLearning:', rewardsQLearning)

    # print('Using Q-learning, total reward over training=',np.sum(rewardsQLearning))
    qlearning_policy = convert_action_values_into_policy(
        stateActionValues)
    qlearning_rewards = run_several_episodes(env, qlearning_policy,
                                             max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                             num_episodes=num_episodes)
    average_reward = np.mean(qlearning_rewards)
    stddev_reward = np.std(qlearning_rewards)
    print('\nUsing Q-learning policy, average reward=',
          average_reward, ' standard deviation=', stddev_reward)

    print('Check the Q-learning policy:')
    pretty_print_policy(env, qlearning_policy)

    if not output_files_prefix == None:
        with open(output_files_prefix + '_optimal.txt', 'w') as f:
            f.write(str(optimal_rewards) + "\n")

        with open(output_files_prefix + '_qlearning.txt', 'w') as f:
            f.write(str(qlearning_rewards) + "\n")

        print("Wrote files", output_files_prefix + "_optimal.txt",
              "and", output_files_prefix + "_qlearning.txt.")


def test_dealing_with_sparsity():
    '''
    When p[s,a,s'] is sparse, one should use
    compute_optimal_state_values
    instead of compute_optimal_state_values_nonsparse,
    and compute_optimal_action_values instead of
    compute_optimal_action_values_nonsparse
    '''
    S = 3
    A = 2
    env = RandomKnownDynamicsEnv(S, A)
    state_values, iteration = compute_optimal_state_values_nonsparse(env)
    print("state_values=", state_values, 'using ', iteration, 'iterations')
    state_values2, iteration = compute_optimal_state_values(env)
    print("state_values with sparsity=", state_values2,
          'using ', iteration, 'iterations')
    assert np.array_equal(state_values, state_values2)

    action_values, stopping_criteria = compute_optimal_action_values_nonsparse(
        env)
    iteration = stopping_criteria.shape[0]
    stopping_criterion = stopping_criteria[-1]
    print("action_values=", action_values, 'using ', iteration, 'iterations')
    print('Stopping criteria until convergence =', stopping_criteria)

    action_values2, stopping_criteria = compute_optimal_action_values(
        env)
    iteration = stopping_criteria.shape[0]
    stopping_criterion = stopping_criteria[-1]
    print("action_values with sparsity=", action_values2, "converged with",
          iteration, "iterations with stopping criterion=", stopping_criterion)
    assert np.array_equal(action_values, action_values2)


def TODO_estimate_model_probabilities(env: gym.Env):
    '''
    # AK-TODO
    Estimate dynamics using Monte Carlo.
    '''
    pass


if __name__ == '__main__':

    #test_dealing_with_sparsity()

    values = np.array([[3, 5, -4, 2], [10, 10, 0, -20]])
    policy = convert_action_values_into_policy(values)
    print("values =", values)
    print("policy =", policy)

    trajectory = np.array([1, 2, 3, 4, 5, 6])
    print("trajectory as an array=", trajectory)
    print("formatted trajectory:")
    print_trajectory(trajectory)
    # test_with_sparse_NextStateProbabilitiesEnv()
    # test_with_NextStateProbabilitiesEnv()

    env = RecycleRobotEnv()
    print("About environment:")
    print("Num of states =", env.S)
    print("Num of actions =", env.A)
    print("env.possible_actions_per_state =", env.possible_actions_per_state)
    print("env.nextStateProbability =", env.nextStateProbability)
    print("env.rewardsTable =", env.rewardsTable)

    uniform_policy = get_uniform_policy_for_known_dynamics(env)
    # several actions (a whole "trajectory")
    num_steps = 10
    taken_actions, rewards_tp1, states = generate_trajectory(
        env, uniform_policy, num_steps)
    trajectory = format_trajectory_as_single_array(
        taken_actions, rewards_tp1, states)
    print("Complete trajectory vector:")
    print(trajectory)
    print("Interpret trajectory with print_trajectory() method:")
    print_trajectory(trajectory)

    print("compute_optimal_state_values(env)=", compute_optimal_state_values(env))

    exit(-1)
    total_rewards = run_episode(
        env, uniform_policy, maxNumIterations=8, printInfo=True, printPostProcessingInfo=True)
    print("Total_rewards =", total_rewards)

    stateActionValues, rewardsQLearning = q_learning_several_episodes(
        env, max_num_time_steps_per_episode=20, episodes_per_run=5000, num_runs=20, verbosity=1)
    print("Q learning stateActionValues =", stateActionValues)
    print("rewardsQLearning =", rewardsQLearning)
