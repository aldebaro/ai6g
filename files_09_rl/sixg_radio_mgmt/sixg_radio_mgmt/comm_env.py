from typing import Callable, Optional, Tuple, Type, Union

import gym
import numpy as np
import yaml

from .basestations import Basestations
from .channel import Channel
from .metrics import Metrics
from .mobility import Mobility
from .slices import Slices
from .traffic import Traffic
from .ues import UEs


class CommunicationEnv(gym.Env):
    """
    Communication environment class using Gym

    ...

    Attributes
    ----------
    variables from config file in env_config/
        Variables characterizing the simulation scenario with information
        for basestations, slices and UEs
    mobility_size : int
        Considers a 2D UE's movement (axis X and Y)
    debug : bool
        If debug is enabled it performs scheduling and other functions
        input/outputs verification. it should disabled in case you want
        to increase performance (decrease simulation time)
    obs_space_format : Optional[Callable[[dict], Union[np.ndarray, dict]]]
        Function defined by the scheduling agent to format the observation space
    calculate_reward : Optional[Callable[[dict], float]]
        Function defined by the scheduling agent to calculate the reward
    action_format : Callable[[np.ndarray, int, int, np.ndarray], np.ndarray]
        Function defined by the scheduling agent to format actions output
    ChannelClass : Type[Channel]
        Channel class defined in channels/ to be used in the simulation
    TrafficClass : Type[Traffic]
        Traffic class defined in traffics/ to be used in the simulation
    MobilityClass : Type[Mobility]
        Mobility class defined in mobilities/ to be used in the simulation
    observation_space : Optional[Callable]
        Function defined by the agent with the observation space using Gym spaces
    action_space: Optional[Callable]
        Function defined by the agent with the action space using Gym spaces

    Methods
    -------
    step(self, sched_decision: np.ndarray)
        Perform one time step in the Gym environment, performing the buffer
        dynamics for each UE, basestation and slice
    reset(self, initial_episode: int = -1)
        Reset the environment when the episode ends
    calculate_reward_default(obs_space: dict)
        Default reward calculation in case the scheduling agent does not
        define it
    obs_space_format_default(obs_space: dict)
        Default observation space formatter in case the scheduling agent
        does not define it
    check_env_agent( self, sched_decision: np.ndarray,
    spectral_efficiencies: np.ndarray, mobilities: np.ndarray,
    traffics: np.ndarray)
        Perform environment and agents verifications to check if everything
        is working as expected (actived with self.debug=True)
    create_scenario(self)
        Create UEs, slices and basestations in accordance with the config
        file used (in env_config/)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ChannelClass: Type[Channel],
        TrafficClass: Type[Traffic],
        MobilityClass: Type[Mobility],
        config_file: str,
        rng: np.random.Generator = np.random.default_rng(),
        action_format: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        obs_space_format: Optional[Callable[[dict], Union[np.ndarray, dict]]] = None,
        calculate_reward: Optional[Callable[[dict], float]] = None,
        obs_space: Optional[Callable] = None,
        action_space: Optional[Callable] = None,
        debug: bool = True,
    ) -> None:
        """initializing the environment.

        Reading config file information and creating channels, mobilities,
        traffics, UEs, basestations and slices.

        Parameters
        ----------
        ChannelClass : Type[Channel]
            Channel class defined in channels/ to be used in the simulation
        TrafficClass : Type[Traffic]
            Traffic class defined in traffics/ to be used in the simulation
        MobilityClass : Type[Mobility]
            Mobility class defined in mobilities/ to be used in the simulation
        config_file : str
            Config file name in env_config/, e.g., "simple"
        action_format : Callable[[np.ndarray, int, int, np.ndarray], np.ndarray]
            Function defined by the scheduling agent to format actions output
        obs_space_format : Optional[Callable[[dict], Union[np.ndarray, dict]]]
            Function defined by the scheduling agent to format the observation space
        calculate_reward : Optional[Callable[[dict], float]]
            Function defined by the scheduling agent to calculate the reward
        observation_space : Optional[Callable]
            Function defined by the agent with the observation space using Gym spaces
        action_space: Optional[Callable]
            Function defined by the agent with the action space using Gym spaces
        debug : bool
            If debug is enabled it performs scheduling and other functions
            input/outputs verification. it should disabled in case you want
            to increase performance (decrease simulation time)
        """
        with open("./env_config/{}.yml".format(config_file)) as file:
            data = yaml.safe_load(file)

        self.max_number_basestations = data["basestations"]["max_number_basestations"]
        self.bandwidths = np.array(data["basestations"]["bandwidths"])  # In MHz
        self.carrier_frequencies = np.array(
            data["basestations"]["carrier_frequencies"]
        )  # In GHz
        self.num_available_rbs = np.array(data["basestations"]["num_available_rbs"])
        self.init_basestation_ue_assoc = np.array(
            data["basestations"]["basestation_ue_assoc"]
        )
        self.init_basestation_slice_assoc = np.array(
            data["basestations"]["basestation_slice_assoc"]
        )

        self.max_number_slices = data["slices"]["max_number_slices"]
        self.init_slice_ue_assoc = np.array(data["slices"]["slice_ue_assoc"])
        self.slice_req = data["slices"]["slice_req"]

        self.max_number_ues = data["ues"]["max_number_ues"]
        self.max_buffer_latencies = np.array(data["ues"]["max_buffer_latencies"])
        self.max_buffer_pkts = np.array(data["ues"]["max_buffer_pkts"])
        self.pkt_sizes = np.array(data["ues"]["pkt_sizes"])  # In bits

        self.step_number = 0  # Initial simulation step
        self.episode_number = 1  # Initial episode
        self.max_number_steps = data["simulation"][
            "max_number_steps"
        ]  # Maximum number of steps per simulated episode
        self.max_number_episodes = data["simulation"][
            "max_number_episodes"
        ]  # Maximum number of simulated episodes
        self.hist_root_path = data["simulation"]["hist_root_path"]
        self.simu_name = data["simulation"]["simu_name"]
        self.associations = data["associations"]
        self.mobility_size = 2  # X and Y axis
        self.debug = debug
        self.rng = rng

        self.obs_space_format = (
            obs_space_format
            if obs_space_format is not None
            else self.obs_space_format_default
        )
        self.calculate_reward = (
            calculate_reward
            if calculate_reward is not None
            else self.calculate_reward_default
        )
        self.action_format = (
            action_format if action_format is not None else self.action_format_default
        )
        self.ChannelClass = ChannelClass
        self.TrafficClass = TrafficClass
        self.MobilityClass = MobilityClass

        if obs_space is not None:
            self.observation_space = obs_space()
        if action_space is not None:
            self.action_space = action_space()

        self.create_scenario()

    def step(
        self, sched_decision: np.ndarray
    ) -> Tuple[Union[np.ndarray, dict], float, bool, dict]:
        """Apply the sched_decision obtained from agent in the environment.

        sched_decisions is a matrix with dimensions BxNxM where B represents
        the number of basestations, N represents the maximum number of UEs
        and M the maximum number of RBs. For instance
        [[[1,1,0], [0,0,1]], [[0,0,1], [1,1,0]]] means that in the basestation
        1, the UE 1 received the RBs 1 and 2 allocated while the second UE
        received the RB 3. For basestation 2, the UE 1 received the RB 3, and
        UE 2 got RBs 1 and 2. Remember that N and M value varies in according
        to the basestation configuration.

        Parameters
        ----------
        sched_decision : np.ndarray
            An array containing all radio resouces allocation for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the evaluated basestation.

        Returns
        -------
        Tuple[Union[np.ndarray, dict], float, bool, dict]
            Tuple containing observation space, reward, end of episode
            bool and human info.
        """

        sched_decision = self.action_format(sched_decision)

        mobilities = self.mobility.step(self.step_number, self.episode_number)
        spectral_efficiencies = self.channel.step(
            self.step_number, self.episode_number, mobilities
        )
        traffics = self.traffic.step(self.step_number, self.episode_number)

        if self.debug:
            self.check_env_agent(
                sched_decision,
                spectral_efficiencies,
                mobilities,
                traffics,
            )
        step_hist = self.ues.step(
            sched_decision,
            traffics,
            spectral_efficiencies,
            self.basestations.bandwidths,
            self.num_available_rbs,
        )
        step_hist.update(
            {
                "mobility": mobilities,
                "spectral_efficiencies": spectral_efficiencies,
                "basestation_ue_assoc": self.basestations.ue_assoc,
                "basestation_slice_assoc": self.basestations.slice_assoc,
                "slice_ue_assoc": self.slices.ue_assoc,
                "sched_decision": sched_decision,
                "slice_req": self.slices.requirements,
            }
        )
        self.step_number += 1
        obs = self.obs_space_format(step_hist)
        reward = self.calculate_reward(step_hist)

        step_hist.update({"reward": reward})
        self.metrics_hist.step(step_hist)

        if self.step_number in self.associations["timeline"]:
            idx_timeline = np.equal(
                self.step_number, self.associations["timeline"]
            ).nonzero()[0][0]
            self.slices.update_assoc(self.associations["slice_ue_assoc"][idx_timeline])
            self.basestations.update_assoc(
                slice_assoc=self.associations["basestation_slice_assoc"][idx_timeline],
                ue_assoc=self.associations["basestation_ue_assoc"][idx_timeline],
            )

        if self.step_number == self.max_number_steps:
            self.metrics_hist.save(self.simu_name, self.episode_number)

        return (
            obs,
            reward,
            self.step_number == self.max_number_steps,
            {},
        )

    def reset(self, initial_episode: int = -1) -> Union[np.ndarray, dict]:
        """Reset the environment.

        in case initial_episode is different from -1, it sets the
        evaluated episode to initial_episode value. It creates a
        scenario again with basestations, slices and UEs.

        Parameters
        ----------
        initial_episode : int
            in case it is different from -1, it sets the evaluated episode
            to initial_episode. It is useful in the test processing to
            define in which episode the test should begins

        Returns
        -------
        Union[np.ndarray, dict]
            Tuple containing observation space after reset the environment
        """
        if (
            (self.step_number == 0 and self.episode_number == 1)
            or (self.episode_number == self.max_number_episodes)
            or initial_episode != -1
        ):
            self.episode_number = 1 if initial_episode == -1 else initial_episode
        elif self.episode_number < self.max_number_episodes:
            self.episode_number += 1
        else:
            raise Exception(
                "Episode number received a non expected value equals to {}.".format(
                    self.episode_number
                )
            )
        self.step_number = 0

        self.create_scenario()
        initial_positions = self.mobility.step(self.step_number, self.episode_number)
        obs = {
            "mobility": initial_positions,
            "spectral_efficiencies": self.channel.step(
                self.step_number, self.episode_number, initial_positions
            ),
            "basestation_ue_assoc": self.basestations.ue_assoc,
            "basestation_slice_assoc": self.basestations.slice_assoc,
            "slice_ue_assoc": self.slices.ue_assoc,
            "sched_decision": [],
            "pkt_incoming": self.traffic.step(self.step_number, self.episode_number),
            "pkt_throughputs": np.zeros(self.max_number_ues),
            "pkt_effective_thr": np.zeros(self.max_number_ues),
            "buffer_occupancies": np.zeros(self.max_number_ues),
            "buffer_latencies": np.zeros(self.max_number_ues),
            "dropped_pkts": np.zeros(self.max_number_ues),
            "slice_req": self.slices.requirements,
        }

        return self.obs_space_format(obs)

    def set_agent_functions(
        self,
        obs_space_format: Optional[Callable[[dict], Union[np.ndarray, dict]]] = None,
        action_format: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        calculate_reward: Optional[Callable[[dict], float]] = None,
    ):
        self.obs_space_format = (
            obs_space_format if obs_space_format is not None else self.obs_space_format
        )
        self.calculate_reward = (
            calculate_reward if calculate_reward is not None else self.calculate_reward
        )
        self.action_format = (
            action_format if action_format is not None else self.action_format
        )

    @staticmethod
    def calculate_reward_default(obs_space: dict) -> float:
        """Default function to calculate reward in case the agent does not define it.

        Parameters
        ----------
        obs_space : dict
            Dictionary with information from environment in the step

        Returns
        -------
        float
            Default reward value
        """
        return 0

    @staticmethod
    def action_format_default(action: np.ndarray) -> np.ndarray:
        return action

    @staticmethod
    def obs_space_format_default(obs_space: dict) -> Union[np.ndarray, dict]:
        """Default function to format the observation space in case the
        agent does not define it.

        Parameters
        ----------
        obs_space : dict
            Dictionary with information from environment in the step

        Returns
        -------
        Union[np.ndarray, dict]
            Default observation space format
        """
        return np.array(list(obs_space.items()), dtype=object)

    def check_env_agent(
        self,
        sched_decision: np.ndarray,
        spectral_efficiencies: np.ndarray,
        mobilities: np.ndarray,
        traffics: np.ndarray,
    ) -> None:
        """Perform environment and agents verifications to check if everything
        is working as expected (actived with self.debug=True)

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
        mobilities: np.ndarray
            Numpy array containing the positions of the UEs in the system
            with shape Ux2, where U represents the maximum number of UEs
            in the system and 2 represents a 2D coordinate of the UE in
            the scenario
        traffics : np.ndarray
            Array containing throughput traffic for each UE (receive packets
            in the buffer)
        """
        # Scheduling decision check
        assert len(sched_decision) == self.max_number_basestations and isinstance(
            sched_decision, np.ndarray
        ), "Sched decision shape does not match the number of basestations or is not of type list"
        for i, basestation_sched in enumerate(sched_decision):
            basestation_sched = np.array(basestation_sched)
            assert basestation_sched.shape == (
                self.max_number_ues,
                self.num_available_rbs[i],
            ), "Scheduling decision does not present the correct shape"
            if np.sum(np.sum(basestation_sched, axis=0) > 1) > 0:
                raise Exception(
                    "Scheduling decision allocated the same RB for more than one UE"
                )
        # Spectral efficiency check
        assert isinstance(
            spectral_efficiencies, np.ndarray
        ), "Spectral efficiencies are not list type"
        for i, basestation_spec in enumerate(spectral_efficiencies):
            assert basestation_spec.shape == (
                self.max_number_ues,
                self.num_available_rbs[i],
            ), "Spectral efficiences have wrong shape."
        # Mobility check
        assert isinstance(mobilities, np.ndarray) and mobilities.shape == (
            self.max_number_ues,
            self.mobility_size,
        ), "Mobility values are not numpy arrays or have wrong shape."
        # Traffics check
        assert isinstance(traffics, np.ndarray) and traffics.shape == (
            self.max_number_ues,
        ), "Traffics values are not numpy arrays or have wrong shape."

    def create_scenario(self) -> None:
        """Create a new scenario with basestation, slices and UEs in
        accordance with initial values.
        """
        self.ues = UEs(
            self.max_number_ues,
            self.max_buffer_latencies,
            self.max_buffer_pkts,
            self.pkt_sizes,
        )
        self.slices = Slices(
            self.max_number_slices,
            self.max_number_ues,
            self.init_slice_ue_assoc,
            self.slice_req,
        )
        self.basestations = Basestations(
            self.max_number_basestations,
            self.max_number_slices,
            self.init_basestation_slice_assoc,
            self.init_basestation_ue_assoc,
            self.bandwidths,
            self.carrier_frequencies,
            self.num_available_rbs,
        )
        self.mobility = self.MobilityClass(self.max_number_ues, rng=self.rng)
        self.channel = self.ChannelClass(
            self.max_number_ues,
            self.max_number_basestations,
            self.num_available_rbs,
            rng=self.rng,
        )
        self.traffic = self.TrafficClass(self.max_number_ues, rng=self.rng)
        self.metrics_hist = Metrics(self.hist_root_path)
