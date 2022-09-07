from typing import Union

import numpy as np
from gym import spaces
from stable_baselines3.sac.sac import SAC

from sixg_radio_mgmt import Agent, CommunicationEnv


class RLSimple(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        hyperparameters: dict = {},
        seed: int = np.random.randint(1000),
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs, seed
        )
        self.agent = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log="./tensorboard-logs/",
            seed=self.seed,
        )

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        return self.agent.predict(np.asarray(obs_space), deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(total_timesteps=int(total_timesteps), callback=[])

    def save(self, filename: str) -> None:
        self.agent.save(filename)

    def load(self, filename: str, env: CommunicationEnv) -> None:
        self.agent = SAC.load(filename, env=env)

    @staticmethod
    def obs_space_format(obs_space: dict) -> np.ndarray:
        formatted_obs_space = np.array([])
        hist_labels = [
            # "pkt_incoming",
            "dropped_pkts",
            # "pkt_effective_thr",
            "buffer_occupancies",
            # "spectral_efficiencies",
        ]
        for hist_label in hist_labels:
            if hist_label == "spectral_efficiencies":
                formatted_obs_space = np.append(
                    formatted_obs_space,
                    np.squeeze(np.sum(obs_space[hist_label], axis=2)),
                    axis=0,
                )
            else:
                formatted_obs_space = np.append(
                    formatted_obs_space, obs_space[hist_label], axis=0
                )

        return formatted_obs_space

    @staticmethod
    def calculate_reward(obs_space: dict) -> float:
        reward = -np.sum(obs_space["dropped_pkts"])
        return reward

    @staticmethod
    def get_action_space() -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(2 * 2 * 2,))

    @staticmethod
    def get_obs_space() -> spaces.Box:
        return spaces.Box(low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64)

    @staticmethod
    def action_format(
        action: np.ndarray,
    ) -> np.ndarray:
        action = np.reshape(action, (2, 2, 2))
        sched_decision = np.copy(action)
        sched_decision[0, 0] = (action[0, 0] >= action[0, 1]).astype(int)
        sched_decision[0, 1] = (action[0, 0] < action[0, 1]).astype(int)
        sched_decision[1, 0] = (action[1, 0] >= action[1, 1]).astype(int)
        sched_decision[1, 1] = (action[1, 0] < action[1, 1]).astype(int)

        return sched_decision
