'''
Generate CSV dataset for ML.
QAM over crazy (non-AWGN) channel.
AK. May 2021
'''
import numpy as np
import commpy.modulation as cm
import commpy.utilities as cu
import matplotlib.pyplot as plt
import commpy.channels
from commpy.channels import SISOFlatChannel
from numpy.random import randn, random, standard_normal

def crazy_channel_propagate(x, snr_dB):
    snr_linear = 10**(0.1*snr_dB)
    signal_power = np.mean(np.abs(x)**2)
    noise_power = signal_power/snr_linear
    noise_std = np.sqrt(noise_power)
    num_samples = x.shape[0]
    y = np.zeros(x.shape)
    gaussian_noise = (standard_normal(num_samples) + 1j * standard_normal(num_samples)) * noise_std * 0.5
    noise = np.abs(x)*0.01 + gaussian_noise
    y = x * noise + 1/x

    return y

def generate_symbols(transmissions=100, M=16):
    """
    Parameters
    ----------
    transmissions: int
        Number of transmissions. Default is 100.
    M: int
        Number of symbols in the constellation. Default is 16.
    Returns
    -------
    """
    bits_per_QAMsymbol = int(np.log2(M))
    bitarrays = [cu.dec2bitarray(obj, bits_per_QAMsymbol)
                 for obj
                 in np.arange(0, M)]
    #print(bitarrays)
    qammod = cm.QAMModem(M)
    const = []
    for i in range(len(bitarrays)):
        const.append(qammod.modulate(bitarrays[i]))
    # unit average power
    constellation = const / np.sqrt((M - 1) * (2 ** 2) / 6)

    ind = np.random.randint(M, size=[1, transmissions])

    # QAM symbols for each antenna
    x = constellation[ind]
    #Reshape to (num_elements,) instead of (num_elements,1)
    #https://www.semicolonworld.com/question/59886/numpy-vector-n-1-dimension-gt-n-dimension-conversion
    x = x.reshape((-1,))
    ind = ind.reshape((-1,))
    return x, ind


def main():
    num_of_symbols = 3000
    symbs, indices=generate_symbols(transmissions=num_of_symbols, M=16)

    SNR_dB=15

    #transmit over the channel
    channel_output = crazy_channel_propagate(symbs, SNR_dB)

    plt.plot(np.real(channel_output),np.imag(channel_output),'bo')
    plt.ylabel('Quadrature')
    plt.xlabel('In-phase')

    #print(indices.shape)
    #if want to plot original Tx symbols
    #for i in range(num_of_symbols):
    #    print(np.real(symbs[i]), ',',np.imag(symbs[i]),',',indices[i])

    plt.plot(np.real(symbs),np.imag(symbs),'rx')

    for i in range(num_of_symbols):
        print(np.real(channel_output[i]), ',',np.imag(channel_output[i]),',',indices[i])


    plt.show()

if __name__ == '__main__':
    main()

