'''
Generate CSV dataset for ML.
QAM over AWGN channel.
AK. May 2021
'''
import numpy as np
import commpy.modulation as cm
import commpy.utilities as cu
import matplotlib.pyplot as plt
import commpy.channels
from commpy.channels import SISOFlatChannel

def qam_constellation(M, unitAvgPower=True):
    bits_per_QAMsymbol = int(np.log2(M))
    bitarrays = [cu.dec2bitarray(obj, bits_per_QAMsymbol)
                 for obj
                 in np.arange(0, M)]
    qammod = cm.QAMModem(M)
    const  = np.array([complex(qammod.modulate(bits)) for bits in bitarrays])

    if unitAvgPower:
        const = const / np.sqrt((M - 1) * (2 ** 2) / 6)

    return const

def qam_demod(x, M, unitAvgPower=True):
    const = qam_constellation(M, unitAvgPower=unitAvgPower)

    const = const.reshape(const.shape[0], 1)
    return abs(x - const).argmin(0)

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
    constellation = qam_constellation(M, unitAvgPower=True)

    ind = np.random.randint(M, size=transmissions)

    # QAM symbols for each antenna
    x   = constellation[ind]

    return x, ind


def main():
    num_of_symbols = 3000
    symbs, indices=generate_symbols(transmissions=num_of_symbols, M=16)
    channel = SISOFlatChannel(None, (1 + 0j, 0j))

    SNR_dB=15  #AK: this is not working!!! SNR does not change
    code_rate=1 #Rate of the used code
    Es=1 #Average symbol energy
    channel.set_SNR_dB(SNR_dB, float(code_rate), Es)

    #transmit over the channel
    channel_output = channel.propagate(symbs)

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

