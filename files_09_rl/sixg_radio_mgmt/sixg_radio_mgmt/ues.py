import numpy as np

from .buffer import Buffer


class UEs:
    """
    Class to implement a UEs functions.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation
    max_buffer_latencies : int
        Maximum latency that packets can wait in the buffer
    max_buffer_pkts: int
        Maximum number of packets allowed in the buffer
    pkt_sizes: np.ndarray
        Array containing the packet size per UE
    buffers: list
        List containing one Buffer object per UE

    Methods
    -------
    get_pkt_throughputs(sched_decision: np.ndarray,
    spectral_efficiencies: np.ndarray, bandwidth: float,
    num_available_rbs: int, pkt_sizes: np.ndarray)
        Calculate packets throughput given the allocation decision,
        spectral efficiences, bandwidth and packet sizes
    def update_ues(self, ue_indexes: np.ndarray,
    max_buffer_latencies: np.ndarray, max_buffer_pkts: np.ndarray,
    pkt_sizes: np.ndarray)
        Update UEs characteristics
    step(self, sched_decision: np.ndarray, traffics: np.ndarray,
    spectral_efficiencies: np.ndarray, bandwidths: np.ndarray,
    num_available_rbs: np.ndarray)
        Give a step in time in which each UEs receive/send packets
        from their buffers
    """

    def __init__(
        self,
        max_number_ues: int,
        max_buffer_latencies: np.ndarray,
        max_buffer_pkts: np.ndarray,
        pkt_sizes: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_buffer_latencies : int
            Maximum latency that packets can wait in the buffer
        max_buffer_pkts: int
            Maximum number of packets allowed in the buffer
        pkt_sizes: np.ndarray
            Array containing the packet size per UE
        """
        self.max_number_ues = max_number_ues
        self.max_buffer_latencies = max_buffer_latencies
        self.max_buffer_pkts = max_buffer_pkts
        self.pkt_sizes = pkt_sizes
        self.buffers = [
            Buffer(max_buffer_pkts[i], max_buffer_latencies[i])
            for i in np.arange(max_number_ues)
        ]

    @staticmethod
    def get_pkt_throughputs(
        sched_decision: np.ndarray,
        spectral_efficiencies: np.ndarray,
        bandwidth: float,
        num_available_rbs: int,
        pkt_sizes: np.ndarray,
    ) -> np.ndarray:
        """Calculate packet throughput for each UE.

        Based on the scheduling decision, spectral efficiences, bandwidth,
        number of available resource blocks and packet sizes calculates
        the throughput for each UE.

        Parameters
        ----------
        sched_decision : np.ndarray
            An array containing all radio resouces allocation for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the evaluated basestation.
        spectral_efficiences : np.ndarray
            An array with the same size of sched_decision containing the
            spectral efficiences values of each pair of UE-Resource block
            in the evaluated basestation
        bandwith : float
            Bandwidth value for the evaluated basestation
        num_available_rbs : int
            Number of available resource blocks in the evaluated basestation
        pkt_sizes : np.ndarray
            Number of resource blocks available in the evaluated basestation

        Returns
        -------
        numpy.ndarray
            An array containing throughput values for all UEs
        """
        return np.floor(
            np.sum(
                (bandwidth / num_available_rbs)
                * np.array(sched_decision)
                * np.array(spectral_efficiencies),
                axis=1,
            )
            / pkt_sizes
        )

    def update_ues(
        self,
        ue_indexes: np.ndarray,
        max_buffer_latencies: np.ndarray,
        max_buffer_pkts: np.ndarray,
        pkt_sizes: np.ndarray,
    ) -> None:
        """Update UEs information.

        Update UEs information in UEs with index in ue_indexes. It generates
        new buffers considering the arguments characteristics.

        Parameters
        ----------
        ue_indexes : np.ndarray
            Array with indexes of the UEs to be changed
        max_buffer_latencies : int
            Maximum latency that packets can wait in the buffer
        max_buffer_pkts: int
            Maximum number of packets allowed in the buffer
        pkt_sizes: np.ndarray
            Array containing the packet size per UE
        """
        self.max_buffer_latencies[ue_indexes] = max_buffer_latencies
        self.max_buffer_pkts[ue_indexes] = max_buffer_pkts
        self.pkt_sizes[ue_indexes] = pkt_sizes
        for ue_index in ue_indexes:
            self.buffers[ue_index] = Buffer(
                max_buffer_pkts[ue_index], max_buffer_latencies[ue_index]
            )

    def step(
        self,
        sched_decision: np.ndarray,
        traffics: np.ndarray,
        spectral_efficiencies: np.ndarray,
        bandwidths: np.ndarray,
        num_available_rbs: np.ndarray,
    ) -> dict:
        """Give a step in time for all UEs.

        Receive packets for all UEs in accordance with traffic values received,
        add them to buffer, after send packets from buffer in accordance with
        the throughput calculated in the get_pkt_throughputs function.

        Parameters
        ----------
        sched_decision : np.ndarray
            An array containing all radio resouces allocation for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the evaluated basestation.
        traffics : np.ndarray
            Array containing throughput traffic for each UE (receive packets
            in the buffer)
        spectral_efficiences : np.ndarray
            An array with the same size of sched_decision containing the
            spectral efficiences values of each pair of UE-Resource block
            in the evaluated basestation
        bandwiths : np.ndarray
            Bandwidth values for the basestations
        num_available_rbs : np.ndarray
            Number of available resource blocks in the basestations

        Returns
        -------
        dict
            Dictionary containing UEs step information
        """
        pkt_throughputs = np.zeros(self.max_number_ues)
        for basestation in np.arange(len(sched_decision)):
            pkt_throughputs += self.get_pkt_throughputs(
                sched_decision[basestation],
                spectral_efficiencies[basestation],
                bandwidths[basestation],
                num_available_rbs[basestation],
                self.pkt_sizes,
            )
        pkt_incomings = np.floor(traffics / self.pkt_sizes)

        for i in np.arange(self.max_number_ues):
            self.buffers[i].receive_packets(pkt_incomings[i])
            self.buffers[i].send_packets(pkt_throughputs[i])
        return {
            "pkt_incoming": pkt_incomings,
            "pkt_throughputs": pkt_throughputs,
            "pkt_effective_thr": np.array(
                [buffer.sent_packets for buffer in self.buffers]
            ),
            "buffer_occupancies": np.array(
                [buffer.get_buffer_occupancy() for buffer in self.buffers]
            ),
            "buffer_latencies": np.array(
                [buffer.get_avg_delay() for buffer in self.buffers]
            ),
            "dropped_pkts": np.array(
                [buffer.dropped_packets for buffer in self.buffers]
            ),
        }
