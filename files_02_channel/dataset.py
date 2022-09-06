import hdf5storage


def data():
    return hdf5storage.loadmat("./files_02_channel/random_sparse_channel_1000_8_8.mat")
