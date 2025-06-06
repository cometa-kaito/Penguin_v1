"""
Gymnasium ラッパ
・任意の env_kwargs を受け取り PenguinPartyEnv に渡せるようにした
・action_masks() で「全 False」になった場合、index 0 (skip) を強制 True
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from penguin_party_env import PenguinPartyEnv

class PenguinPartyGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, **env_kwargs):
        super().__init__()
        self.env = PenguinPartyEnv(**env_kwargs)

        self.action_map = self._generate_action_map()
        self.action_to_index = {act: i for i, act in enumerate(self.action_map)}
        self.action_space = spaces.Discrete(len(self.action_map))

        self.observation_space = spaces.Dict({
            "hand":  spaces.MultiDiscrete([6] * 14),
            "board": spaces.Box(low=0, high=5, shape=(7, 13), dtype=np.uint8),
        })

    # Gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._encode_obs(obs), {}

    def step(self, action_idx: int):
        action = self.action_map[int(action_idx)]
        obs, reward, done, info = self.env.step(action)

        if obs is None:
            obs = {"hand": [], "board": {}, "current_player": 0}

        encoded = self._encode_obs(obs)
        return encoded, reward, done, False, info

    def render(self):
        self.env.render()

    # Maskable PPO ------------------------------------------------------
    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for act in self.env.get_valid_actions():
            mask[self.action_to_index[act]] = True
        if not mask.any():          # 保険：必ず 1 個は True
            mask[0] = True
        return mask

    # 内部ユーティリティ ----------------------------------------------
    def _generate_action_map(self):
        colors = self.env.colors
        actions = [(None, None)]  # index 0 = skip
        for row in range(1, 8):
            for col in range(1, 14):
                for c in colors:
                    actions.append((f"{row}-{col}", c))
        return actions

    def _encode_obs(self, obs: dict):
        board_array = np.zeros((7, 13), dtype=np.uint8)
        for pos, color in obs["board"].items():
            row, col = map(int, pos.split("-"))
            board_array[row - 1, col - 1] = self.env.colors.index(color) + 1

        hand_ids = [self.env.colors.index(c) + 1 for c in obs["hand"]]
        hand_ids += [0] * (14 - len(hand_ids))
        return {"hand": np.array(hand_ids, dtype=np.int64), "board": board_array}
