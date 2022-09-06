from .channel_gen import ChannelGenerator, normalize
from .channel_conversion import antenna
from .mimo_channels import ak_generate_sparse_channels

import numpy as np


class RandomSparseChannelGenerator(ChannelGenerator):
    def __init__(self, Nr, Nt, num_clusters, rand_gen: np.random.Generator):
        self.Nr = Nr
        self.Nt = Nt
        self.num_clusters = num_clusters
        self.rand_gen = rand_gen

    def __next__(self):
        Hv = ak_generate_sparse_channels(
            self.num_clusters, self.Nr, self.Nt, self.rand_gen
        )
        Hv = normalize(Hv)
        H = antenna(Hv)

        return H, Hv

    def shape(self):
        return (self.Nr, self.Nt)
