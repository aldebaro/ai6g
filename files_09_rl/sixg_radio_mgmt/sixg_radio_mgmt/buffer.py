import numpy as np


class Buffer:
    """
    Implement buffer functions following a FIFO architecture. Each UE has
    a buffer.

    ...

    Attributes
    ----------
    buffer : np.ndarray
        Array representing the number of packets that waited in the buffer
        in each time step n, with maximum time step equals to the maximum
        packet latency allowed by max_packet_age variable
    cumulative_buffer : np.ndarray
        Auxiliary variable to calculate sent packets
    max_packets_buffer: int
        Maximum number of packets allowed in the buffer
    max_packets_age : int
        Maximum latency that packets can wait in the buffer
    dropped_packets : int
        Accounts the number of dropped packets in a step
    sent_packets : int
        Accounts the number of sent packets in a step

    Methods
    -------
    receive_packets(self, num_packets_arrived: int)
        Add arrived packets to the buffer structure
    send_packets(self, packets_available_to_send: int)
        Send buffer packets (free buffer space)
    get_buffer_occupancy(self)
        Get the buffer occupancy rate
    get_avg_delay(self)
        Get the average delay of buffer packets
    """

    def __init__(self, max_packets_buffer: int, max_packet_age: int) -> None:
        """
        Parameters
        ----------
        max_packets_buffer: int
            Maximum number of packets allowed in the buffer
        max_packets_age : int
            Maximum latency that packets can wait in the buffer
        """
        self.buffer = np.zeros(max_packet_age + 1)
        self.cumulative_buffer = np.zeros(max_packet_age + 1)
        self.max_packets_buffer = max_packets_buffer
        self.max_packets_age = max_packet_age
        self.dropped_packets = 0
        self.sent_packets = 0

    def receive_packets(self, num_packets_arrived: int) -> None:
        """Add arrived packets in the buffer.

        The buffer is represented by a 1-D array, where the index 0
        represents the packets that arrived now and the last index n
        represents the packets which are waiting n steps to be transmitted.
        A packet is dropped when it stays for more than max_packet_age
        steps or when the num_packets_arrived is greater than the buffer
        space available (depends on max_packets_buffer).

        Parameters
        ----------
        num_packets_arrived: int
            Number of packets that arrived in the buffer
        """
        self.dropped_packets = 0
        self.dropped_packets += self.buffer[-1]
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = 0
        if (np.sum(self.buffer) + num_packets_arrived) <= self.max_packets_buffer:
            self.buffer[0] = num_packets_arrived
        else:
            self.dropped_packets += num_packets_arrived - (
                self.max_packets_buffer - np.sum(self.buffer)
            )
            self.buffer[0] = self.max_packets_buffer - np.sum(self.buffer)

    def send_packets(self, packets_available_to_send: int) -> None:
        """Send packets from available buffer.

        Transmit packets from buffer to free buffer space. It allocates the packets
        waiting longer time (near from last array element) first.

        Parameters
        ----------
        packets_available_to_send: int
            Number of packets to be sent from buffer
        """
        tmp_buffer = self.buffer.copy()
        if (self.get_buffer_occupancy() != 0) or (packets_available_to_send != 0):
            for i in np.arange(self.buffer.shape[0])[::-1]:
                if packets_available_to_send >= self.buffer[i]:
                    packets_available_to_send -= self.buffer[i]
                    self.buffer[i] = 0
                else:
                    self.buffer[i] -= packets_available_to_send
                    break
        self.cumulative_buffer += np.subtract(tmp_buffer, self.buffer)
        self.sent_packets = np.sum(tmp_buffer) - np.sum(self.buffer)

    def get_buffer_occupancy(self) -> float:
        """Get buffer occupancy rate.

        Returns
        -------
        float
            Buffer occupancy rate
        """
        return np.sum(self.buffer) / self.max_packets_buffer

    def get_avg_delay(self) -> float:
        """Get average buffer delay.

        Returns
        -------
        float
            Average buffer delay
        """
        if np.sum(self.cumulative_buffer) != 0:
            return np.sum(
                self.cumulative_buffer * np.arange(self.max_packets_age + 1)
            ) / np.sum(self.cumulative_buffer)
        else:
            return 0


def main():
    # Testing Buffer functions
    arrived_packets = [0, 10, 0, 5, 2, 2, 10, 15, 20, 10]
    sent_packets = 2
    ue_buffer = Buffer(20, 5)
    for packets in arrived_packets:
        ue_buffer.receive_packets(packets)
        print(
            ue_buffer.buffer,
            ",  Number of dropped packets: ",
            ue_buffer.dropped_packets,
            ", Number of sent packets: ",
            ue_buffer.sent_packets,
        )
        ue_buffer.send_packets(sent_packets)
        print(ue_buffer.buffer, "\n")

    print(ue_buffer.cumulative_buffer)
    print("UE Average delay: ", ue_buffer.get_avg_delay(), " ms")


if __name__ == "__main__":
    main()
