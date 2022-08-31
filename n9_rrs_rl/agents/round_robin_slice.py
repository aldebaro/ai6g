from typing import Union

import numpy as np

from sixg_radio_mgmt import Agent, CommunicationEnv


class RoundRobin(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )

    def step(self, obs_space: Union[dict, np.ndarray]) -> np.ndarray:
        allocation_rbs = [
            np.zeros((self.max_number_ues, self.num_available_rbs[basestation]))
            for basestation in np.arange(self.max_number_basestations)
        ]
        # Below is a fixed inter-slice allocation in which slice 1 always
        # receives more resources than slice 2
        slice_1_rate = 3 / 4
        slice_2_rate = 1 / 4
        for basestation in np.arange(self.max_number_basestations):
            rb_pos = 0
            for slice in np.arange(len(obs_space["slice_ue_assoc"])):
                if obs_space["basestation_slice_assoc"][basestation][slice] == 1:
                    num_slice_available_rbs = (
                        self.num_available_rbs[basestation] * slice_1_rate
                        if slice == 0
                        else self.num_available_rbs[basestation] * slice_2_rate
                    )
                    ues_slice_basestation_assoc = (
                        obs_space["slice_ue_assoc"][slice]
                        * obs_space["basestation_ue_assoc"][basestation]
                    )
                    rbs_per_ue = int(
                        np.floor(
                            num_slice_available_rbs
                            / np.sum(ues_slice_basestation_assoc)
                        )
                    )
                    for idx, ue in enumerate(ues_slice_basestation_assoc):
                        if ue == 1:
                            allocation_rbs[basestation][
                                idx, rb_pos : rb_pos + rbs_per_ue
                            ] = 1
                            rb_pos += rbs_per_ue

        return np.array(allocation_rbs)

    @staticmethod
    def obs_space_format(obs_space: dict) -> dict:
        return obs_space

    @staticmethod
    def calculate_reward(obs_space: dict) -> float:
        return 0

    def action_format(self, action: np.ndarray) -> np.ndarray:
        return action
