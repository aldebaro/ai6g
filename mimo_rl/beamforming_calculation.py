'''
Assumes ULA at base station and single-antenna users.
Shows the grid world using pygame.
Globecom Tutorial - December 7, 2021
Tutorial 29: Machine Learning for MIMO Systems with Large Arrays
Nuria Gonzalez-Prelcic (NCSU),
Aldebaro Klautau (UFPA) and
Robert W. Heath Jr. (NCSU)
'''
import numpy as np

class AnalogBeamformer:
    def __init__(self, num_antenna_elements=32):
        self.num_antenna_elements = num_antenna_elements
        self.codebook = dft_codebook(self.num_antenna_elements)
        num_points = 500  # resolution for plotting the beams
        # dimension num_points x num_antenna_elements
        self.beams_for_plotting, self.angles_for_plotting = self.get_steered_factors(
            num_antenna_elements, num_points)

    def get_steered_factors(self, num_antenna_elements, num_points):
        # Calculate the steering factor
        # grid, in angular domain
        theta = np.linspace(-np.pi, np.pi, num_points)
        theta = theta[:, np.newaxis]
        arrayFactors = arrayFactorGivenAngleForULA(
            num_antenna_elements, theta)
        steeredArrayFactors = np.squeeze(
            np.matmul(arrayFactors, self.codebook))
        return steeredArrayFactors, theta

    def get_num_codevectors(self):
        return self.codebook.shape[0]

    def get_best_beam_index(self, H):
        EquivalentChannels = np.dot(np.squeeze(np.asarray(H)), self.codebook)
        bestIndex = np.argmax(np.abs(EquivalentChannels))
        return bestIndex

    def get_combined_channel(self, beam_index, channel_h):
        combined_channel = np.dot(np.squeeze(
            np.asarray(channel_h)), self.codebook[:,beam_index])
        combined_channel = float(np.abs(combined_channel))
        return combined_channel

def arrayFactorGivenAngleForULA(self, numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):
    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    arrayFactor = arrayFactor / np.sqrt(numAntennaElements)
    return arrayFactor  # normalize to have unitary norm

#creates a DFT codebook
def dft_codebook(dim):
    seq = np.matrix(np.arange(dim))
    mat = seq.conj().T * seq
    w = np.exp(-1j * 2 * np.pi * mat / dim)
    return w

def getNarrowBandULAMIMOChannel(azimuths_tx, azimuths_rx, p_gainsdB, number_Tx_antennas, number_Rx_antennas,
                                normalizedAntDistance=0.5, angleWithArrayNormal=0, pathPhases=None):
    """
    - assumes one beam per antenna element

    the first column will be the elevation angle, and the second column is the azimuth angle correspondingly.
    p_gain will be a matrix size of (L, 1)
    departure angle/arrival angle will be a matrix as size of (L, 2), where L is the number of paths

    t1 will be a matrix of size (nt, nr), each
    element of index (i,j) will be the received
    power with the i-th precoder and the j-th
    combiner in the departing and arrival codebooks
    respectively

    :param departure_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param arrival_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param p_gaindB: path gain (L, 1) in dB where L is the number of paths
    :param number_Rx_antennas, number_Tx_antennas: number of antennas at Rx and Tx, respectively
    :param pathPhases: in degrees, same dimension as p_gaindB
    :return:
    """
    azimuths_tx = np.deg2rad(azimuths_tx)
    azimuths_rx = np.deg2rad(azimuths_rx)
    # nt = number_Rx_antennas * number_Tx_antennas #np.power(antenna_number, 2)
    m = np.shape(azimuths_tx)[0]  # number of rays
    H = np.matrix(np.zeros((number_Rx_antennas, number_Tx_antennas)))

    gain_dB = p_gainsdB
    path_gain = np.power(10, gain_dB / 10)
    path_gain = np.sqrt(path_gain)

    # generate uniformly distributed random phase in radians
    if pathPhases is None:
        pathPhases = 2*np.pi * np.random.rand(len(path_gain))
    else:
        # convert from degrees to radians
        pathPhases = np.deg2rad(pathPhases)

    # include phase information, converting gains in complex-values
    path_complexGains = path_gain * np.exp(-1j * pathPhases)

    # recall that in the narrowband case, the time-domain H is the same as the
    # frequency-domain H
    for i in range(m):
        # at and ar are row vectors (using Python's matrix)
        at = np.matrix(arrayFactorGivenAngleForULA(number_Tx_antennas, azimuths_tx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        ar = np.matrix(arrayFactorGivenAngleForULA(number_Rx_antennas, azimuths_rx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        # outer product of ar Hermitian and at
        H = H + path_complexGains[i] * ar.conj().T * at
    factor = (np.linalg.norm(path_complexGains) / np.sum(path_complexGains)) * np.sqrt(
        number_Rx_antennas * number_Tx_antennas)  # scale channel matrix
    H *= factor  # normalize for compatibility with Anum's Matlab code

    return H


def arrayFactorGivenAngleForULA(numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):
    '''
    Calculate array factor for ULA for angle theta. If angleWithArrayNormal=0
    (default),the angle is between the input signal and the array axis. In
    this case when theta=0, the signal direction is parallel to the array
    axis and there is no energy. The maximum values are for directions 90
        and -90 degrees, which are orthogonal with array axis.
    If angleWithArrayNormal=1, angle is with the array normal, which uses
    sine instead of cosine. In this case, the maxima are for
        thetas = 0 and 180 degrees.
    References:
    http://www.waves.utoronto.ca/prof/svhum/ece422/notes/15-arrays2.pdf
    Book by Balanis, book by Tse.
    '''
    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(
            1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    arrayFactor = arrayFactor / np.sqrt(numAntennaElements)
    return arrayFactor  # normalize to have unitary norm

if __name__ == '__main__':
    analogBeamformer = AnalogBeamformer()
    print('# codevectors=', analogBeamformer.get_num_codevectors())
    beam_index = 5
    channel_h = np.ones((1,32))
    print('gain mag=',analogBeamformer.get_combined_channel(beam_index, channel_h))
