'''
We adopt bidirectional maps based on https://pypi.org/project/bidict/
'''
import numpy as np
import itertools
from bidict import bidict

def get_position_combinations(grid_size,Nu):
    #positions: we are restricted to square M x M grids
    positions_x_axis = np.arange(grid_size)
    all_positions_single_user = list(itertools.product(positions_x_axis, repeat=2))
    all_positions = list(itertools.product(all_positions_single_user, repeat=Nu))
    return all_positions

def convert_list_of_possible_tuples_in_bidict(list_of_tuples):
    #assume there are no repeated elements
    N = len(list_of_tuples)
    this_bidict = bidict()
    for n in range(N):
        this_bidict.put(n,list_of_tuples[n])
    return this_bidict

if __name__ == '__main__':
    x=list()
    y=np.zeros((3,1))
    #x.append(y) #does not work because unhashable type: 'numpy.ndarray'
    x.append((3,5,'a'))
    x.append((3,4,'a'))
    x.append(('b'))
    bidict = convert_list_of_possible_tuples_in_bidict(x)
    print(bidict[1])
    print(bidict.inv['b'])

    #test return
    x = np.random.randn(3,4)
    print(x)
    a = tuple(x.flatten())
    print(a)
    print(len(a))
    b = np.array(a).reshape((3,4))
    print(b)
    exit(-1)
