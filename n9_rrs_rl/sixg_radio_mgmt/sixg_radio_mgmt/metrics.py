import os
from typing import Dict

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    """
    Metrics class to save/load metrics from simulations.

    ...

    Attributes
    ----------
    root_path : str
        Root path to save/load simulations
    metrics_hist : dict
        Variable to save/load simulations metric values

    Methods
    -------
    step(self, hist: dict)
        Append metric values to the metrics_hist variable
    save(self, simu_name: str, episode_number: int)
        Save metric values obtained over a simulation to an external file
    read(root_path: str, simu_name: str, episode_number: int)
        Read external files containing metrics from previous simulations
    """

    def __init__(
        self,
        root_path: str,
    ) -> None:
        """
        Parameters
        ----------
        root_path : str
            Root path to save/load simulations
        """
        self.root_path = root_path
        self.metrics_hist = {
            "pkt_incoming": [],
            "pkt_throughputs": [],
            "pkt_effective_thr": [],
            "buffer_occupancies": [],
            "buffer_latencies": [],
            "dropped_pkts": [],
            "mobility": [],
            "spectral_efficiencies": [],
            "basestation_ue_assoc": [],
            "basestation_slice_assoc": [],
            "slice_ue_assoc": [],
            "sched_decision": [],
            "reward": [],
            "slice_req": [],
        }

    def step(self, hist: dict) -> None:
        """Append metric values to local variable metrics_hist

        Parameters
        ----------
        hist : dict
            Metric values obtained in the last simulation step
        """
        for metric in hist.keys():
            self.metrics_hist[metric].append(hist[metric])

    def save(self, simu_name: str, episode_number: int) -> None:
        """Save collected metric values to an external file

        Parameters
        ----------
        simu_name : str
            Simulation name to compose the directory name to save results
        episode_number: int
            Episode number being evaluated in the simulation to compose
            external file name
        """
        path = ("{}/hist/{}/").format(
            self.root_path,
            simu_name,
        )
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed(path + "ep_" + str(episode_number), **self.metrics_hist)

    @staticmethod
    def read(root_path: str, simu_name: str, episode_number: int) -> Dict:
        """Read external file containing metric results to a dict.

        Parameters
        ----------
        root_path : str
            Root path to save/load simulations
        simu_name : str
            Simulation name to compose the directory name to save results
        episode_number: int
            Episode number being evaluated in the simulation to compose
            external file name

        Returns
        -------
        dict
            Dictionary containing all the metric values for the requested
            simulation and episode number
        """
        path = "{}/hist/{}/ep_{}.npz".format(root_path, simu_name, episode_number)
        data = np.load(path, allow_pickle=True)
        data_dict = {
            "pkt_incoming": data.f.pkt_incoming,
            "pkt_throughputs": data.f.pkt_throughputs,
            "pkt_effective_thr": data.f.pkt_effective_thr,
            "buffer_occupancies": data.f.buffer_occupancies,
            "buffer_latencies": data.f.buffer_latencies,
            "dropped_pkts": data.f.dropped_pkts,
            "mobility": data.f.mobility,
            "spectral_efficiencies": data.f.spectral_efficiencies,
            "basestation_ue_assoc": data.f.basestation_ue_assoc,
            "basestation_slice_assoc": data.f.basestation_slice_assoc,
            "slice_ue_assoc": data.f.slice_ue_assoc,
            "sched_decision": data.f.sched_decision,
            "reward": data.f.reward,
            "slice_req": data.f.slice_req,
        }
        return data_dict


class Plots:
    @staticmethod
    def plot(
        xlabel: str,
        x_data: np.ndarray,
        ylabel: str,
        y_data: np.ndarray,
        y_data_label: np.ndarray,
        fig_name: str,
        metric: str,
    ) -> None:
        w, h = matfig.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid()
        for i, y in enumerate(np.arange(y_data.shape[1])):
            plt.plot(x_data, y_data[:, i], label=y_data_label[i])
        fig.tight_layout()
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/{}_{}.pdf".format(fig_name, metric),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        plt.close()


def main():
    data = Metrics.read("./", "test", 1)

    metrics = [
        "pkt_incoming",
        "pkt_throughputs",
        "pkt_effective_thr",
        "buffer_occupancies",
        "buffer_latencies",
        "dropped_pkts",
    ]
    # traffics
    for metric in metrics:
        labels = np.array(
            ["ue {}".format(i) for i in np.arange(1, data[metric].shape[1] + 1)]
        )
        Plots.plot(
            "Step n",
            np.arange(data[metric].shape[0]),
            "Packets",
            data[metric],
            labels,
            "test",
            metric,
        )


if __name__ == "__main__":
    main()
