"""
MIMO processing
"""
import numpy as np


def exppdf(x, mu):
    return np.exp(-x / mu) / mu


def ak_generate_sparse_channels(
    num_clusters, Nr, Nt, rand_gen: np.random.Generator, tau_sigma=1e-9, mu=0.2
):
    """
    ########################################################################################
    # Modified from
    # From: Author: Anum Ali
    #
    # If you use this code or any (modified) part of it in any publication, please cite
    # the paper: Anum Ali, Nuria González-Prelcic and Robert W. Heath Jr.,
    # "Millimeter Wave Beam-Selection Using Out-of-Band Spatial Information",
    # IEEE Transactions on Wireless Communications.
    #
    # Contact person email: anumali@utexas.edu
    ########################################################################################
    # Input Arguments:
    # tau_sigma: The RMS delay spread of the channel
    # mu: The exponential PDF parameter
    # num_clusters: The number of clusters
    # num_ant_bs_yaxis and num_ant_bs_xaxis: num of antennas at Rx and Tx
    tau_sigma=3e-9; #seconds
    mu=0.2;
    num_clusters=4;
    num_ant_bs_yaxis=4;
    num_ant_bs_xaxis=4;
    ########################################################################################
    """
    # find different delays

    taus = tau_sigma * np.log(
        rand_gen.random(size=(num_clusters,))
    )  # beautiful transformation
    taus = np.sort(taus - np.min(taus))[::-1]  # synchronize Rx with first impulse
    # PDP = stats.expon.pdf(
    #     taus / tau_sigma, scale=mu
    # )  # Exponential PDP. Matlab: PDP=exppdf(taus/tau_sigma,mu)
    PDP = exppdf(taus / tau_sigma, mu)
    PDP = PDP / np.sum(PDP)  # Normalizing to unit power PDP
    gains = np.sqrt(PDP / 2 / num_clusters) * (
        rand_gen.standard_normal((num_clusters, 2)).view(dtype="complex128").squeeze()
    )
    Hv = np.zeros((Nr, Nt), dtype=complex)
    num_H_elements = Nr * Nt
    # choose without replacement
    chosen_indices = rand_gen.choice(
        num_H_elements, num_clusters, replace=False
    )  # Matlab's randsample
    for i in range(num_clusters):
        chosen_index_uraveled = np.unravel_index(chosen_indices[i], Hv.shape)
        Hv[chosen_index_uraveled] = gains[i]
    return Hv
