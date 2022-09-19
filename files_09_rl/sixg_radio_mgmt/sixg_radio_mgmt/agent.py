from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .comm_env import CommunicationEnv


class Agent(ABC):
    """
    Agent abstract class to implement a radio resource scheduling method.

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
    step(self, obs_space: Union[np.ndarray, dict])
        Abstract method to define the allocation of radio resources for UEs
        based on the observation space
    obs_space_format(obs_space: dict)
        Abstract and static method to format the observation space variable
        in a form to be recognized by the agent
    calculate_reward(obs_space: dict)
        Abstract and static method to calculate the reward to the agent
    action_format(action: np.ndarray, max_number_ues: int,
                    max_number_basestations: int,
                    num_available_rbs: np.ndarray)
        Abstract and static method to put action output in a format allowed
        by the environemnt
    """

    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        seed: int = np.random.randint(1000),
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
        self.env = env
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.num_available_rbs = num_available_rbs
        self.seed = seed

    @abstractmethod
    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        """Decide the radio resource allocation.

        Based on the observation space variable defines the radio resource
        allocation in accordance with the adopted policy

        Parameters
        ----------
        obs_space : np.ndarray or dict
            The observation space variables containing the simulation
            environment information

        Returns
        -------
        numpy.ndarray
            An array containing all radio resouces allocation for each UE
            in all basestations
        """
        pass

    @abstractmethod
    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        """Format the obsevation space to the agent's step function

        Format the observation space variable in a form to be recognized
        by the agent, e.g., in a RL environment using Gym interface, the
        train method needs to receive a list/array to proceed with training
        process, so the obs_space_format should return a numpy array type.

        Parameters
        ----------
        obs_space : np.ndarray or dict
            The observation space variables containing the simulation
            environment information

        Returns
        -------
        np.ndarray or a dict
            Formatted observation space to step function
        """
        pass

    @abstractmethod
    def calculate_reward(self, obs_space: dict) -> float:
        """Calculate the reward

        Based on the observation space calculates the reward value.

        Parameters
        ----------
        obs_space : dict
            The observation space variables containing the simulation
            environment information

        Returns
        -------
        float
            Reward value
        """
        pass

    @abstractmethod
    def action_format(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """Format action variable

        Format the action variable to a format compatible with the
        simulated environment.

        Parameters
        ----------
        action : np.ndarray
            Numpy array containing the action (allocation decision) taken
            by the agent step function
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_number_basestations : int
            Maximum number of basestations in the simulation
        num_available_rbs : np.ndarray
            Number of radio resource blocks available per basestation

        Returns
        -------
        np.ndarray
            An array containing all radio resouces allocation for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the given basestation.
        """
        return action
