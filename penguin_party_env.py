"""
Penguin Party 環境（2 人・14 枚・5 色）
・合法手が 1 つも無くなった時点でゲーム終了とする実装に変更
・get_valid_actions() は常に 1 件以上（最悪 [(None, None)]）を返す
"""

import random
from typing import Dict, List, Tuple, Optional

class PenguinPartyEnv:
    # ──────────────────────────────────────────────────────────────────
    #  初期化
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        num_players: int = 2,
        max_hand_size: int = 14,
        colors: Optional[List[str]] = None,
        deck_distribution: Optional[Dict[str, int]] = None,
        reward_config: Optional[Dict[str, float]] = None,
        initial_position_policy: str = "center-only",
    ):
        self.num_players   = num_players
        self.max_hand_size = max_hand_size
        self.colors        = colors or ["Green", "Red", "Blue", "Yellow", "Purple"]

        self.deck_distribution = deck_distribution or {
            "Green": 8, "Red": 7, "Blue": 7, "Yellow": 7, "Purple": 7,
        }

        self.default_reward = {
            "valid_action_bonus":     0.05,
            "win_base":               1.0,
            "win_per_diff":           0.1,
            "lose_base":             -1.0,
            "lose_per_diff":          0.1,
            "invalid_action_penalty":-0.2,
        }
        self.reward_config = self.default_reward if reward_config is None else reward_config

        assert initial_position_policy in ("center-only", "free")
        self.initial_position_policy = initial_position_policy

        self.reset()

    # ──────────────────────────────────────────────────────────────────
    #  基本 API
    # ──────────────────────────────────────────────────────────────────
    def reset(self):
        deck: List[str] = []
        for c, n in self.deck_distribution.items():
            deck.extend([c] * n)
        random.shuffle(deck)

        self.hands: List[List[str]] = [
            [deck.pop() for _ in range(self.max_hand_size)]
            for _ in range(self.num_players)
        ]
        self.board: Dict[str, str] = {}
        self.current_player: int = 0
        self.done: bool = False
        self.skipped: set[int] = set()
        return self._get_observation()

    def step(self, action: Tuple[Optional[str], Optional[str]]):
        if self.done or self.current_player is None:
            return self._get_observation(), 0.0, True, {"info": "game done"}

        # 合法手が 0 → 即終了（マスク全 False を防ぐ）
        valid_actions = self.get_valid_actions()
        if valid_actions == [(None, None)]:
            self.done = True
            scores = self.get_scores()
            reward = self._calc_end_reward(self.current_player, scores)
            return None, reward, True, {"info": "no legal move → game over"}

        position, color = action

        # ---------- 無効手 ----------
        if (position, color) not in valid_actions:
            self.skipped.add(self.current_player)
            next_player = self._next_active_player()
            self.current_player = next_player
            if next_player is None:
                self.done = True
            return (
                self._get_observation(),
                self.reward_config["invalid_action_penalty"],
                self.done,
                {"info": "invalid action"},
            )

        # ---------- スキップ ----------
        if position is None and color is None:
            self.skipped.add(self.current_player)
            if len(self.skipped) == self.num_players:
                self.done = True
                scores = self.get_scores()
                reward = self._calc_end_reward(self.current_player, scores)
                return None, reward, True, {"info": "all skipped (game over)"}
            self.current_player = self._next_active_player()
            return self._get_observation(), 0.0, False, {"info": "skip"}

        # ---------- 正常手 ----------
        self.board[position] = color
        self.hands[self.current_player].remove(color)

        if all(len(h) == 0 for h in self.hands):
            self.done = True
        next_player = self._next_active_player()
        if next_player is None:
            self.done = True

        reward = self.reward_config["valid_action_bonus"]
        if self.done:
            scores = self.get_scores()
            reward += self._calc_end_reward(self.current_player, scores)

        self.current_player = next_player if not self.done else self.current_player
        return self._get_observation(), reward, self.done, {}

    # ──────────────────────────────────────────────────────────────────
    #  補助メソッド
    # ──────────────────────────────────────────────────────────────────
    def _get_observation(self):
        if self.done or self.current_player is None:
            return None
        return {
            "hand": list(self.hands[self.current_player]),
            "board": dict(self.board),
            "current_player": self.current_player,
        }

    def _next_active_player(self) -> Optional[int]:
        start = self.current_player
        for i in range(1, self.num_players + 1):
            cand = (start + i) % self.num_players
            if cand not in self.skipped:
                return cand
        return None

    # 合法手生成 --------------------------------------------------------
    def is_valid_action(self, position: str, color: str) -> bool:
        if position in self.board or color not in self.hands[self.current_player]:
            return False
        try:
            row, col = map(int, position.split("-"))
        except ValueError:
            return False
        if not (1 <= row <= 7 and 1 <= col <= 13):
            return False

        # 最下段
        if row == 1:
            base_cols = sorted(int(k.split("-")[1]) for k in self.board if k.startswith("1-"))
            if not base_cols:
                return col == 7 if self.initial_position_policy == "center-only" else True
            if len(base_cols) >= 7:
                return False
            return col == base_cols[0] - 1 or col == base_cols[-1] + 1

        # 上段
        left = f"{row-1}-{col}"
        right = f"{row-1}-{col+1}"
        if left not in self.board or right not in self.board:
            return False
        return color in (self.board[left], self.board[right])

    def get_valid_actions(self):
        return self.get_valid_actions_for(self.current_player)

    def get_valid_actions_for(self, player_idx: int):
        if player_idx is None or player_idx >= self.num_players:
            return [(None, None)]
        valid: List[Tuple[Optional[str], Optional[str]]] = []
        hand = self.hands[player_idx]
        for row in range(1, 8):
            for col in range(1, 14):
                pos = f"{row}-{col}"
                for c in set(hand):
                    if self.is_valid_action(pos, c):
                        valid.append((pos, c))
        if not valid:
            valid.append((None, None))
        return valid

    # スコア・報酬 ------------------------------------------------------
    def get_scores(self):
        return [len(h) for h in self.hands]

    def _calc_end_reward(self, current_player: int, scores: List[int]) -> float:
        my_score = scores[current_player]
        opp_scores = [s for i, s in enumerate(scores) if i != current_player]
        if not opp_scores:
            return 0.0
        opp_min = min(opp_scores)
        diff = opp_min - my_score
        if my_score < opp_min:
            return self.reward_config["win_base"] + self.reward_config["win_per_diff"] * diff
        elif my_score > opp_min:
            return self.reward_config["lose_base"] + self.reward_config["lose_per_diff"] * diff
        return 0.0
