#penguin_party_gym.py
"""
Gymnasium ラッパ
・action_masks() で「全 False」になった場合、index 0 (skip) を強制 True
・step()でエピソード終了時は両者の最終報酬をinfo["final_rewards"]から取得可能
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
            "action_mask": spaces.MultiBinary(self.action_space.n),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        enc = self._encode_obs(obs)
        enc["action_mask"] = self.action_masks()
        return enc, {}

    def step(self, action_idx: int):
        action = self.action_map[int(action_idx)]
        obs, reward, done, info = self.env.step(action)

        if obs is None:
            obs = {"hand": [], "board": {}, "current_player": 0}

        encoded = self._encode_obs(obs)
        encoded["action_mask"] = self.action_masks()

        # エピソード終了時は両者の最終報酬をinfo["final_rewards"]から参照可能
        if done and "final_rewards" in info:
            encoded["final_rewards"] = info["final_rewards"]

        return encoded, reward, done, False, info

    def render(self):
        self.env.render()

    def action_masks(self):
        if self.env.done or self.env.current_player is None:
            return np.zeros(self.action_space.n, dtype=bool)

        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_actions = self.env.get_valid_actions()

        for act in valid_actions:
            idx = self.action_to_index.get(act)
            if idx is not None:
                mask[idx] = True
            else:
                print(f"[ERROR] action_masks: action {act} not found in action_to_index.")

        if not mask.any():
            print("[CRITICAL] action_masks: No valid actions found; forcing action[0]=True.")
            mask[0] = True

        return mask

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
