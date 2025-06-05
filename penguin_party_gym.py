import numpy as np
import gymnasium as gym
from gymnasium import spaces
from penguin_party_env import PenguinPartyEnv

class PenguinPartyGymEnv(gym.Env):
    """
    PenginPartyEnv → Gymnasium ラッパ
    --------------------------------------------------
    ・行動空間 : Discrete (skip + 7×13×色)
    ・観測空間 : Dict(hand[14], board[7×13])
      - hand  : 0=空, 1–5=色ID+1
      - board : 0=空, 1–5=色ID+1
    ・MaskablePPO 対応の action_masks() 実装済
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.env = PenguinPartyEnv()   # 元環境インスタンス

        # -------- 行動マップ作成 --------
        self.action_map = self._generate_action_map()            # index → (pos, color)
        self.action_to_index = {act: i for i, act in enumerate(self.action_map)}
        self.action_space = spaces.Discrete(len(self.action_map))

        # -------- 観測空間定義 --------
        # hand : MultiDiscrete([6]*14) → 0–5 (5 色 + 空)
        # board: Box((7,13), uint8)    → 0–5
        self.observation_space = spaces.Dict({
            "hand":  spaces.MultiDiscrete([6] * 14),
            "board": spaces.Box(low=0, high=5, shape=(7, 13), dtype=np.uint8),
        })

    # ------------------------------------------------------------------ #
    #  Gym API 実装
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        """環境リセット → Gym 仕様の (obs, info) を返す"""
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._encode_obs(obs), {}

    def step(self, action_idx: int):
        """
        Parameters
        ----------
        action_idx : int
            Discrete 空間の整数。self.action_map を介して (pos, color) に変換。
        Returns
        -------
        obs, reward, terminated, truncated, info : Gymnasium フォーマット
        """
        action = self.action_map[int(action_idx)]
        obs, reward, done, info = self.env.step(action)

        if obs is None:                     # 終了後はダミー観測を用意
            obs = {"hand": [], "board": {}, "current_player": 0}

        encoded = self._encode_obs(obs)
        terminated = done                   # 時間切れではなく純粋なゲーム終了
        truncated  = False                  # 打ち切り条件は未使用
        return encoded, reward, terminated, truncated, info

    def render(self):
        self.env.render()                   # オリジナル環境のテキスト描画

    # ------------------------------------------------------------------ #
    #  Maskable PPO 用: アクションマスク
    # ------------------------------------------------------------------ #
    def action_masks(self):
        """
        戻り値
        -------
        np.ndarray(bool) : shape=(action_space.n,)
            True の index だけ選択可能 (合法手)
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        for act in self.env.get_valid_actions():
            mask[self.action_to_index[act]] = True
        return mask

    # ------------------------------------------------------------------ #
    #  内部ユーティリティ
    # ------------------------------------------------------------------ #
    def _generate_action_map(self):
        """
        Discrete → (position, color) 対応表を生成。
        index 0 は (None, None) = スキップ固定。
        """
        colors = self.env.colors
        actions = [(None, None)]                        # index 0 = skip
        for row in range(1, 8):                         # 行 1–7
            for col in range(1, 14):                    # 列 1–13
                for c in colors:
                    actions.append((f"{row}-{col}", c))
        return actions

    def _encode_obs(self, obs: dict):
        """
        Dict 観測 → 数値エンコード
        * board: 7×13 の uint8 行列
        * hand : 長さ 14 の uint8 ベクトル
        """
        # --- board エンコード ---
        board_array = np.zeros((7, 13), dtype=np.uint8)
        for pos, color in obs["board"].items():
            row, col = map(int, pos.split("-"))
            board_array[row - 1, col - 1] = self.env.colors.index(color) + 1

        # --- hand エンコード ---
        hand_ids = [self.env.colors.index(c) + 1 for c in obs["hand"]]
        hand_ids += [0] * (14 - len(hand_ids))          # 空スロットは 0

        return {
            "hand":  np.array(hand_ids, dtype=np.int64),
            "board": board_array,
        }
