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

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        allocation_rbs = [
            np.zeros((self.max_number_ues, self.num_available_rbs[basestation]))
            for basestation in np.arange(self.max_number_basestations)
        ]
        for basestation in np.arange(self.max_number_basestations):
            ue_idx = 0
            rb_idx = 0
            while rb_idx < self.num_available_rbs[basestation]:
                if obs_space[basestation][ue_idx] == 1:
                    allocation_rbs[basestation][ue_idx][rb_idx] += 1
                    rb_idx += 1
                ue_idx += 1 if ue_idx + 1 != self.max_number_ues else -ue_idx

        return np.array(allocation_rbs)

    def obs_space_format(self, obs_space: dict) -> np.ndarray:
        return np.array(obs_space["basestation_ue_assoc"])

    def calculate_reward(self, obs_space: dict) -> float:
        reward = -np.sum(obs_space["dropped_pkts"])
        return reward

    def action_format(self, action: np.ndarray) -> np.ndarray:
        return action
