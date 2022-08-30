'''
From: https://github.com/lasseufpa/ml4comm-icc21/

Very simple channel calculation for Mimo_RL_Simple_Env.
We assume the base station has an ULA and the users have
single antennas. The channels are time-invariant, and we
can pre-calculate them. But these channels do not depend
only on the position of the user, but also on the position
of all other users, which are scatterers for the given user.
Hence, for all Npos combinations of users' positions, we
calculate the channels of all Nu users.
Each channel is obtained either using a line-of-sight (LOS)
or non-LOS approximation based on the geometric channel.
'''
import numpy as np
from numpy.random import rand #uniform [0,1)
from numpy.random import randn #Gaussian
import copy
from beamforming_calculation import getNarrowBandULAMIMOChannel
from bidict import bidict
from mimo_rl_tools import get_position_combinations
from mimo_rl_tools import convert_list_of_possible_tuples_in_bidict

#restricted to 2 users
class Grid_Mimo_Channel:

    def __init__(self, num_antenna_elements=32, grid_size=6):
        self.num_bs_antenna_elements = num_antenna_elements
        self.grid_size = grid_size
        self.angle_spread = 1 #angle spread in degrees for LOS channels
        #get all possible positions of a pair of users
        self.Nu = 2 #number of users
        all_positions = get_position_combinations(self.grid_size, self.Nu)
        self.positions_bidict = convert_list_of_possible_tuples_in_bidict(all_positions)
        #construct all channels
        self.use_only_nlos = True #simplified channel generation
        self.all_channels = self.initialize_all_channels()

    def initialize_all_channels(self):
        num_of_positions = len(self.positions_bidict)
        #define as complex-valued channels
        all_channels = np.zeros((self.Nu,num_of_positions, self.num_bs_antenna_elements), dtype=complex)
        for u in range(self.Nu):
            for i in range(num_of_positions):
                if self.use_only_nlos:
                    all_channels[u][i]=self.get_nlos_channel(num_rays = 3)
                else:
                    (x,y) = self.positions_bidict[i]
                    #TODO: loop over get_los_channel()
                    raise NotImplementedError()                    
        return all_channels
    
    def get_nlos_channel(self, num_rays = 3):
        complex_channels = randn(num_rays) + np.random.randn(num_rays)
        gain_in_dB = 20*np.log10(np.abs(complex_channels))  
        phase = np.angle(complex_channels)*180/np.pi #in degrees
        AoA_az = 180.0*np.random.rand(num_rays) #in degrees
        AoD_az = 180.0*np.random.rand(num_rays) #in degrees
        Nr = 1
        h = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, 
            self.num_bs_antenna_elements, Nr, pathPhases=phase)
        return h

    def get_specific_channel(self, positions, user):
        position_index = self.positions_bidict.inv[positions]
        #deep copy to avoid someone changing the values outside this class
        #maybe I should avoid deepcopy for increased speed
        return copy.deepcopy(self.all_channels[user][position_index])

    #TODO
    def get_los_channel():
        raise NotImplementedError()  
        gain_in_dB = friis_propagation(Ptx, distance, freq, gain=gain)
        AoD_az = departure + angle_spread*np.random.randn(numRays)
        AoA_az = arrival + angle_spread*np.random.randn(numRays)
        phase = np.angle(gain)*180/np.pi
            
        Ht = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, Nt, Nr, pathPhases=phase) 
        Ht = Ht / np.linalg.norm(Ht) #normalize channel to unit norm
        return Ht

if __name__ == '__main__':
    mimo_Channel = Grid_Mimo_Channel(num_antenna_elements=32, grid_size=6)
    print('h1.shape=',mimo_Channel.all_channels[0][1].shape)
    print('h2.shape=', mimo_Channel.get_nlos_channel().shape)
    print('h1=',mimo_Channel.all_channels[0][1])
    print('h3=', mimo_Channel.get_nlos_channel())