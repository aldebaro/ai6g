from abc import ABC, abstractmethod

import numpy as np


class Channel(ABC):
    """
    Channel abstract class to implement a channel generator to be used
    in the simulation.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation
    max_number_basestations : int
        Maximum number of basestations in the simulation
    num_available_rbs : np.ndarray
        Number of radio resource blocks available per basestation

    Methods
    -------
    def step(self, step_number: int, episode_number: int,
            mobilities: np.ndarray)
        Abstract method to define the allocation of radio resources for UEs
        based on the observation space
    """

    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_number_basestations : int
            Maximum number of basestations in the simulation
        num_available_rbs : np.ndarray
            Number of radio resource blocks available per basestation
        """
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs
        self.rng = rng

    @abstractmethod
    def step(
        self, step_number: int, episode_number: int, mobilities: np.ndarray
    ) -> np.ndarray:
        """Abstract function to generate channel values per UExRB.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation
        mobilities: np.ndarray
            Numpy array containing the positions of the UEs in the system
            with shape Ux2, where U represents the maximum number of UEs
            in the system and 2 represents a 2D coordinate of the UE in
            the scenario

        Returns
        -------
        np.ndarray
            An array containing all spectral efficiency values for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the given basestation.
        """
        pass
