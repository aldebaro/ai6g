from abc import ABC, abstractmethod

import numpy as np


class Mobility(ABC):
    """
    Mobility abstract class to implement a UEs mobilities.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate 2D positions for each UE in the simulation
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
        """Generate UEs movement in the simulation.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        mobilities: np.ndarray
            Numpy array containing the positions of the UEs in the system
            with shape Ux2, where U represents the maximum number of UEs
            in the system and 2 represents a 2D coordinate of the UE in
            the scenario
        """
        pass
