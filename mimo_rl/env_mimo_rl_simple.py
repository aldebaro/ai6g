'''
Mimo_RL_simple an an OpenAI gym environment
Globecom Tutorial - December 7, 2021
Tutorial 29: Machine Learning for MIMO Systems with Large Arrays
Nuria Gonzalez-Prelcic (NCSU),
Aldebaro Klautau (UFPA) and
Robert W. Heath Jr. (NCSU)
'''
import numpy as np
from numpy.random import randint
import itertools
import gym
from gym import spaces
from beamforming_calculation import AnalogBeamformer
from channel_mimo_rl_simple import Grid_Mimo_Channel
from mimo_rl_tools import convert_list_of_possible_tuples_in_bidict
from mimo_rl_tools import get_position_combinations

class Mimo_RL_Simple_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, num_antenna_elements=32, grid_size=6, should_render=False):
        super(Mimo_RL_Simple_Env, self).__init__()
        self.__version__ = "0.1.0"

        self.analogBeamformer = AnalogBeamformer(num_antenna_elements=num_antenna_elements)
        self.grid_Mimo_Channel = Grid_Mimo_Channel(num_antenna_elements=num_antenna_elements, grid_size=6)
        self.should_render = should_render
        if self.should_render:
            from render_mimo_rl_simple import Mimo_RL_render
            self.mimo_RL_render = Mimo_RL_render(self.analogBeamformer)

        #Use True to represent the state by an integer index, as in a 
        #finite Markov decision process
        self.observation_equals_state_index = True

        self.episode_duration = 30 #number of steps in an episode
        self.penalty = 100 #if not switching among users
        #a single user can be served by the base station
        self.Nu = 2 #number of users
        self.Nb = self.analogBeamformer.get_num_codevectors() #number of beams
        self.Na = 3 #user must be allocated at least once in Na allocations
        #define grid size: grid_size x grid_size
        self.grid_size = grid_size 
        #directions each user takes (left, right, up, down). Chosen at the
        #beginning of each episode
        self.users_directions_indices = np.zeros(self.Nu)
        #directions: up, down, right, left
        self.position_updates = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        self.last_action_index = -1 #last taken action
        self.episode_return = 0 #return: sum of rewards over the episode

        #obstacles, fixed objects, which do not move
        self.obstacles = [[1,2], [3,4], [4,4]]
        #self.Tx = [1,2]
        #self.wall1 = [3,4]
        #self.wall2 = [4,4]

        #We adopt bidirectional maps based on https://pypi.org/project/bidict/
        self.bidict_actions = convert_list_of_possible_tuples_in_bidict(self.get_all_possible_actions())
        self.bidict_states = convert_list_of_possible_tuples_in_bidict(self.get_all_possible_states())
        #self.bidict_rewards = convert_list_of_possible_tuples_in_bidict()
        #I don't need a table for all the rewards. I will generate them as we go. 
        #Otherwise I would implement:
        #def get_all_possible_rewards(self):
        self.reward = 0

        #could get this info from the above bidicts
        self.S = len(self.get_all_possible_states())        
        self.A = len(self.get_all_possible_actions())

        #Define spaces to make environment compatible with the library Stable Baselines
        self.action_space = spaces.Discrete(self.get_num_actions())
        if self.observation_equals_state_index:
            self.observation_space = spaces.Discrete(self.get_num_states())
        else:
            high_value = np.maximum(self.Nu, self.grid_size)-1
            #TODO need to conclude this code
            self.observation_space = spaces.Box(low=0, high=high_value,
                                    shape=( (2*self.Nu + (self.Na-1)),), 
                                    dtype=np.uint8)
        
        #keep current state information based only on its index
        self.current_state_index = 0

        self.currentIteration = 0 #continuous, count time and also TTIs
        self.reset() #create variables

    def step(self, action_index):
        """
        The agent takes a step in the environment.
        """
        #interpret action: convert from index to useful information
        scheduled_user, beam_index = self.interpret_action(action_index)

        #get current state
        positions, previously_scheduled = self.interpret_state(self.current_state_index)

        throughput = self.get_througput(scheduled_user, beam_index)
        self.reward = throughput

        allocated_users = np.array(previously_scheduled)
        allocated_users = np.append(allocated_users, scheduled_user)

        if len(np.unique(allocated_users)) != self.Nu:
            self.reward = throughput - self.penalty
        
        #update for next iteration
        previous_state_index = self.current_state_index
        self.last_action_index = action_index
        #loop to shift to the left
        allocated_users_tuple = tuple(allocated_users[1:])        
        #get new positions for the users. Note that this does not depend
        #on the action taken by the agent
        new_positions = self.update_users_positions(positions) 
        self.current_state_index = self.convert_state_to_index(tuple(new_positions),tuple(allocated_users_tuple))
        self.episode_return += self.reward

        #update renderer
        if self.should_render:
            self.mimo_RL_render.set_positions(positions, scheduled_user, beam_index)
            self.mimo_RL_render.render()

        #check if episode has finished
        gameOver = False
        if self.currentIteration == self.episode_duration:
            ob = self.reset()
            gameOver = True  # game ends
        else:
            ob = self.get_state()
  
        # history version with actions and states
        history = {"time": self.currentIteration,
                   "action_t": self.interpret_action(action_index),
                   "state": self.interpret_state(previous_state_index),
                   "positions": positions,
                   "reward": self.reward,
                   "return": self.episode_return,
                   #"users_directions_indices": self.users_directions_indices,
                   "next_state": self.interpret_state(self.current_state_index)}
        
        self.currentIteration += 1 #update iteration counter
        return ob, self.reward, gameOver, history

    def enable_rendering(self):
        self.should_render = True
        from render_mimo_rl_simple import Mimo_RL_render
        self.mimo_RL_render = Mimo_RL_render(self.analogBeamformer)

    def get_num_states(self):
        return self.S

    def get_num_actions(self):
        return self.A

    def get_current_reward(self):
        return self.reward

    #note that bidict cannot hash numpy arrays. We will use tuples
    def get_all_possible_actions(self):
        '''Nu is the number of users and Nb the number of beam pairs'''
        all_served_users = range(self.Nu)
        list_beam_indices = range(self.Nb)
        all_actions = [(a,b) for a in all_served_users for b in list_beam_indices]
        return all_actions

    #note that bidict cannot hash numpy arrays. We will use tuples
    def get_all_possible_states(self):
        #positions: we are restricted to square M x M grids
        all_positions = get_position_combinations(self.grid_size, self.Nu)        
        #positions_x_axis = np.arange(self.grid_size)
        #positions_y_axis = np.arange(self.grid_size)
        #all_positions_single_user = list(itertools.product(positions_x_axis, repeat=2))
        #all_positions = list(itertools.product(all_positions_single_user, repeat=self.Nu))

        #previously scheduled users
        previously_scheduled = list(itertools.product(np.arange(self.Nu), repeat=self.Na-1))
        #print(previously_scheduled)
        
        all_states = list(itertools.product(all_positions, previously_scheduled))
        #all_states = [(a,b) for a in all_positions for b in previously_scheduled]
        return all_states

    def get_UE_positions(self):
        positions, previously_scheduled = self.interpret_state(self.current_state_index)
        return positions

    def convert_state_to_index(self,positions,previously_scheduled):
        state = (positions, previously_scheduled)
        state_index = self.bidict_states.inv[state]
        return state_index

    def interpret_action(self, action_index):
        action = self.bidict_actions[action_index]
        scheduled_user = action[0]
        beam_index = action[1]
        return scheduled_user, beam_index

    def interpret_state(self, state_index):
        state = self.bidict_states[state_index]
        positions = state[0]
        previously_scheduled = state[1]
        return positions, previously_scheduled

    #recall the directions: up, down, right, left
    #self.position_updates = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    def deal_with_obstacles(self, u, new_position_array):
        num_obstacles = len(self.obstacles)
        for i in range(num_obstacles):
            if (self.obstacles[i][0] == new_position_array[0]) and (self.obstacles[i][1] == new_position_array[1]):
                #go back to previous position
                new_position_array = new_position_array - self.position_updates[self.users_directions_indices[u]]
                #change direction
                if self.users_directions_indices[u] == 0:
                    self.users_directions_indices[u] = 1
                elif self.users_directions_indices[u] == 1:
                    self.users_directions_indices[u] = 0
                elif self.users_directions_indices[u] == 2:
                    self.users_directions_indices[u] = 3
                elif self.users_directions_indices[u] == 3:
                    self.users_directions_indices[u] = 2
                break        
        return new_position_array

    #get new positions, avoiding obstacles and base station
    def update_users_positions(self, positions):
        positions_as_array = np.array(positions)
        new_positions = list()        
        for u in range(self.Nu):
            new_position_array = positions_as_array[u] + self.position_updates[self.users_directions_indices[u]]
            new_position_array = self.deal_with_obstacles(u, new_position_array)
            #wrap-around grid:
            new_position_array[np.where(new_position_array>self.grid_size-1)] = 0
            new_position_array[np.where(new_position_array<0)] = self.grid_size-1
            new_positions.append(tuple(new_position_array))
        return tuple(new_positions)

    #calculate an estimated throughput based on the combined channel
    #associated to the actual channel and precoding vector
    def get_througput(self, scheduled_user, beam_index):
        positions, previously_scheduled = self.interpret_state(self.current_state_index)
        channel_h = self.grid_Mimo_Channel.get_specific_channel(positions, scheduled_user)
        channel_mag = self.analogBeamformer.get_combined_channel(beam_index, channel_h)
        return channel_mag

    def get_state(self):
        """Get the current observation.
        """
        if self.observation_equals_state_index:
            return self.current_state_index
        else:
            return self.bidict_states[self.current_state_index]

    def get_last_action(self):
        return self.interpret_action(self.last_action_index)

    #def render(self, mode='human'):
    def render(self):
        self.mimo_RL_render.render()
    
    def close (self):
        pass

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentIteration = 0
        self.episode_return = 0
        self.users_directions_indices = randint(0,4,size=(self.Nu,))
        self.current_state_index = randint(0, self.get_num_states())
        return self.get_state()

    #This was needed in previous versions but it does not seem needed anymore
    # from gym.utils import seeding
    # def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def numberOfActions(self):
        return self.A

    def numberOfObservations(self):
        return self.S

def run_debug():
    from stable_baselines.common.env_checker import check_env
    num_antenna_elements=32
    grid_size=6
    mimo_RL_Environment = Mimo_RL_Simple_Env(num_antenna_elements=num_antenna_elements, grid_size=grid_size)
    check_env(mimo_RL_Environment)

if __name__ == '__main__':
    num_antenna_elements=32
    grid_size=6
    mimo_RL_Environment = Mimo_RL_Simple_Env(num_antenna_elements=num_antenna_elements, 
        grid_size=grid_size, should_render=True)

    num_steps = 500
    num_actions = mimo_RL_Environment.get_num_actions()
    for i in range(num_steps):
        action_index = randint(0,num_actions)
        ob, reward, gameOver, history = mimo_RL_Environment.step(action_index)
        if gameOver:
            print('Game over! End of episode.')
        print(history)
