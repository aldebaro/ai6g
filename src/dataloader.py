import tensorflow as tf
import numpy as np


class LidarDataset2D(object):
    def __init__(self, lidar_data_path, beam_data_path):
        # this allows us to merge multiple dsets into one
        if isinstance(lidar_data_path, list) and isinstance(beam_data_path, list):
            lidar_data = None
            beam_output = None
            beam_output_true = None
            for lidar_path, beam_path in zip(lidar_data_path, beam_data_path):
                if lidar_data is not None and beam_output is not None:
                    lidar_data = np.concatenate([lidar_data, lidar_to_2d(lidar_path)], axis=0)
                    beam_output = np.concatenate([beam_output, get_beam_output(beam_path)[0]], axis=0)
                    beam_output_true = np.concatenate([beam_output_true, get_beam_output_no_normalization(beam_path, threshold=np.inf)[0]], axis=0)

                else:
                    lidar_data = lidar_to_2d(lidar_path)
                    beam_output = get_beam_output(beam_path)[0]
                    beam_output_true = get_beam_output_no_normalization(beam_path, threshold=np.inf)[0]

        else:
            lidar_data = lidar_to_2d(lidar_data_path)
            if beam_data_path is None:
                beam_output = np.zeros((lidar_data.shape[0], 256))
                beam_output_true = np.zeros((lidar_data.shape[0], 256))
            else:
                beam_output = get_beam_output(beam_data_path)[0]
                beam_output_true = get_beam_output_no_normalization(beam_data_path, threshold=np.inf)[0]

        self.lidar_data = np.expand_dims(lidar_data, 1)
        self.beam_output = beam_output
        self.beam_output_true = beam_output_true


class LidarDataset3D(object):
    def __init__(self, lidar_data_path, beam_data_path):
        # this allows us to merge multiple dsets into one
        if isinstance(lidar_data_path, list) and isinstance(beam_data_path, list):
            lidar_data = None
            beam_output = None
            beam_output_true = None
            for lidar_path, beam_path in zip(lidar_data_path, beam_data_path):
                if lidar_data is not None and beam_output is not None:
                    lidar_data = np.concatenate([lidar_data, np.load(lidar_path)['input']], axis=0)
                    beam_output = np.concatenate([beam_output, get_beam_output(beam_path)[0]], axis=0)
                    beam_output_true = np.concatenate([beam_output_true, get_beam_output_no_normalization(beam_path, threshold=np.inf)[0]], axis=0)
                else:
                    lidar_data = np.load(lidar_path)['input']
                    beam_output = get_beam_output(beam_path)[0]
                    beam_output_true = get_beam_output_no_normalization(beam_path, threshold=np.inf)[0]

        else:
            lidar_data = np.load(lidar_data_path)['input']
            if beam_data_path is None:
                beam_output = np.zeros((lidar_data.shape[0], 256))
                beam_output_true = np.zeros((lidar_data.shape[0], 256))
            else:
                beam_output = get_beam_output(beam_data_path)[0]
                beam_output_true = get_beam_output_no_normalization(beam_data_path, threshold=np.inf)[0]

        self.lidar_data = np.transpose(lidar_data, (0, 3, 1, 2))
        self.beam_output = beam_output
        self.beam_output_true = beam_output_true


def beams_log_scale(y, thresholdBelowMax):
    y_shape = y.shape

    for i in range(0, y_shape[0]):
        thisOutputs = y[i, :]
        logOut = 20 * np.log10(thisOutputs + 1e-30)
        minValue = np.amax(logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs[zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum(thisOutputs)
        y[i, :] = thisOutputs

    return y


def get_beam_output(output_file, threshold=6):
    thresholdBelowMax = threshold

    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    # new ordering of the beams, provided by the Organizers
    y = np.zeros((yMatrix.shape[0], num_classes))
    for i in range(0, yMatrix.shape[0], 1):  # go over all examples
        codebook = np.absolute(yMatrix[i, :])  # read matrix
        Rx_size = codebook.shape[0]  # 8 antenna elements
        Tx_size = codebook.shape[1]  # 32 antenna elements
        for tx in range(0, Tx_size, 1):
            for rx in range(0, Rx_size, 1):  # inner loop goes over receiver
                y[i, tx * Rx_size + rx] = codebook[rx, tx]  # impose ordering

    y = beams_log_scale(y, thresholdBelowMax)

    return y, num_classes


def get_beam_output_no_normalization(output_file, threshold=6):
    thresholdBelowMax = threshold

    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    # new ordering of the beams, provided by the Organizers
    y = np.zeros((yMatrix.shape[0], num_classes))
    for i in range(0, yMatrix.shape[0], 1):  # go over all examples
        codebook = np.absolute(yMatrix[i, :])  # read matrix
        Rx_size = codebook.shape[0]  # 8 antenna elements
        Tx_size = codebook.shape[1]  # 32 antenna elements
        for tx in range(0, Tx_size, 1):
            for rx in range(0, Rx_size, 1):  # inner loop goes over receiver
                y[i, tx * Rx_size + rx] = codebook[rx, tx]  # impose ordering

    return y, num_classes


def lidar_to_2d(lidar_data_path):

    lidar_data = np.load(lidar_data_path)['input']

    lidar_data1 = np.zeros_like(lidar_data)[:, :, :, 1]

    lidar_data1[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_data1[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_data1[np.max(lidar_data == -1, axis=-1)] = -1

    return lidar_data1
