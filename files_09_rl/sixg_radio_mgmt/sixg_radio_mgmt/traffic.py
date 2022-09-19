from abc import ABC, abstractmethod

import numpy as np


class Traffic(ABC):
    """
    Traffic abstract class to implement a UEs traffics.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate throughput traffic for each UE in the simulation
    """

    def __init__(
        self, max_number_ues: int, rng: np.random.Generator = np.random.default_rng()
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        """
        self.max_number_ues = max_number_ues
        self.rng = rng

    @abstractmethod
    def step(self, step_number: int, episode_number: int) -> np.ndarray:
        """Generate UEs traffic in the simulation.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        np.ndarray
            Numpy array containing the throughput traffic of the UEs in
            the system
        """
        pass
