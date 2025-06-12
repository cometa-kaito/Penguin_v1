#!/usr/bin/env python3
"""
penguin_party_env.py
────────────────────────────────────────────────────────
Penguin Party（2 人・14 枚・5 色）簡易環境

* プレイヤー0 が学習対象
* 行動: (position: str | None, color: str | None)
    - 無効手       → invalid_action_penalty
    - スキップ手   → 0
    - 有効手       → per_card_bonus
* 終局時に _calc_final_rewards() の本人分を加算
────────────────────────────────────────────────────────
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Set


class PenguinPartyEnv:
    # ---------- 初期化 ----------
    def __init__(
        self,
        num_players: int = 2,
        max_hand_size: int = 14,
        colors: Optional[List[str]] = None,
        deck_distribution: Optional[Dict[str, int]] = None,
        reward_config: Optional[Dict[str, float]] = None,
        initial_position_policy: str = "center-only",
    ):
        self.num_players: int = num_players
        self.max_hand_size: int = max_hand_size
        self.colors: List[str] = colors or ["Green", "Red", "Blue", "Yellow", "Purple"]

        self.deck_distribution: Dict[str, int] = deck_distribution or {
            "Green": 8,
            "Red": 7,
            "Blue": 7,
            "Yellow": 7,
            "Purple": 7,
        }

        self._default_reward: Dict[str, float] = {
            "per_card_bonus": 0.01,
            "win_base": 10.0,
            "win_per_diff": 10.0,
            "lose_base": -10.0,
            "lose_per_diff": 10.0,
            "invalid_action_penalty": -1.0,
        }
        self.reward_config: Dict[str, float] = (
            self._default_reward
            if reward_config is None
            else {**self._default_reward, **reward_config}
        )

        assert initial_position_policy in ("center-only", "free")
        self.initial_position_policy = initial_position_policy

        # 状態変数
        self.hands: List[List[str]] = []
        self.board: Dict[str, str] = {}
        self.current_player: Optional[int] = 0
        self.done: bool = False
        self.skipped: Set[int] = set()

        self.reset()

    # ---------- ゲーム開始 ----------
    def reset(self):
        """山札をシャッフルし、手札と盤面を初期化"""
        deck: List[str] = []
        for c, n in self.deck_distribution.items():
            deck.extend([c] * n)
        random.shuffle(deck)

        self.hands = [
            [deck.pop() for _ in range(self.max_hand_size)]
            for _ in range(self.num_players)
        ]
        self.board = {}
        self.current_player = 0
        self.done = False
        self.skipped.clear()
        return self._get_observation()

    # ---------- 1 手進める ----------
    def step(self, action: Tuple[Optional[str], Optional[str]]):
        """
        Parameters
        ----------
        action : Tuple[str|None, str|None]
            (position, color) いずれか None でスキップ

        Returns
        -------
        obs, reward, done, info
        """
        if self.done or self.current_player is None:
            return self._get_observation(), 0.0, True, {"info": "game done"}

        position, color = action
        valid_actions = self.get_valid_actions()
        is_agent_turn = self.current_player == 0

        # ---- 無効手 --------------------------------------------------
        if (position, color) not in valid_actions:
            self.skipped.add(self.current_player)
            next_player = self._next_active_player()
            self.current_player = next_player
            if next_player is None:  # 全員退場
                self.done = True
                final_rewards = self._terminal_rewards()
                reward = (
                    self.reward_config["invalid_action_penalty"] if is_agent_turn else 0.0
                ) + final_rewards[0]
                return (
                    self._get_observation(),
                    reward,
                    True,
                    {"info": "invalid action", "final_rewards": final_rewards},
                )
            pen = self.reward_config["invalid_action_penalty"] if is_agent_turn else 0.0
            return self._get_observation(), pen, False, {"info": "invalid action"}

        # ---- スキップ ------------------------------------------------
        if position is None and color is None:
            self.skipped.add(self.current_player)
            next_player = self._next_active_player()
            if next_player is None:  # 全員スキップで終了
                self.done = True
                final_rewards = self._terminal_rewards()
                reward = final_rewards[0]  # agent 固定
                return (
                    self._get_observation(),
                    reward,
                    True,
                    {"info": "all skipped", "final_rewards": final_rewards},
                )
            self.current_player = next_player
            return self._get_observation(), 0.0, False, {"info": "skip"}

        # ---- 正常手 --------------------------------------------------
        self.board[position] = color
        self.hands[self.current_player].remove(color)

        reward = self.reward_config["per_card_bonus"] if is_agent_turn else 0.0

        # 終局判定
        if all(len(h) == 0 for h in self.hands):
            self.done = True
        next_player = self._next_active_player()
        if next_player is None:
            self.done = True

        info = {}
        if self.done:
            final_rewards = self._terminal_rewards()
            reward += final_rewards[0]
            info["final_rewards"] = final_rewards

        self.current_player = next_player if not self.done else self.current_player
        return self._get_observation(), reward, self.done, info

    # ---------- 観測 ----------
    def _get_observation(self):
        if self.done or self.current_player is None:
            return None
        return {
            "hand": list(self.hands[self.current_player]),
            "board": dict(self.board),
            "current_player": self.current_player,
        }

    # ---------- 補助 ----------
    def _next_active_player(self) -> Optional[int]:
        """スキップしていない次プレイヤー番号（全員退場なら None）"""
        start = self.current_player
        for i in range(1, self.num_players + 1):
            cand = (start + i) % self.num_players
            if cand not in self.skipped:
                return cand
        return None

    def _terminal_rewards(self) -> Dict[int, float]:
        return self._calc_final_rewards(self.get_scores())

    # ---------- 盤面ロジック ----------
    def is_valid_action(self, position: str, color: str) -> bool:
        """指定手が合法か判定"""
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
            base_cols = sorted(
                int(k.split("-")[1]) for k in self.board if k.startswith("1-")
            )
            if not base_cols:
                return (
                    col == 7
                    if self.initial_position_policy == "center-only"
                    else True
                )
            if len(base_cols) >= 7:
                return False
            return col == base_cols[0] - 1 or col == base_cols[-1] + 1

        # 上段
        left, right = f"{row-1}-{col}", f"{row-1}-{col+1}"
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

    # ---------- スコア & 終局報酬 ----------
    def get_scores(self) -> List[int]:
        """各プレイヤーの残り手札枚数"""
        return [len(h) for h in self.hands]

    def _calc_final_rewards(self, scores: List[int]) -> Dict[int, float]:
        rewards: Dict[int, float] = {}
        for i in range(self.num_players):
            my_score = scores[i]
            opp_scores = [s for j, s in enumerate(scores) if j != i]
            opp_min = min(opp_scores)
            diff = opp_min - my_score
            if my_score < opp_min:  # 勝ち
                r = self.reward_config["win_base"] + self.reward_config["win_per_diff"] * diff
            elif my_score > opp_min:  # 負け
                r = self.reward_config["lose_base"] + self.reward_config["lose_per_diff"] * diff
            else:  # 引き分け
                r = 0.0
            rewards[i] = r
        return rewards

    # ---------- 追加 API ----------
    def outcome(self, player_idx: int = 0) -> str:
        """'win' / 'lose' / 'draw' の三値を返す"""
        scores = self.get_scores()
        my_score = scores[player_idx]
        opp_best = min(s for i, s in enumerate(scores) if i != player_idx)
        if my_score < opp_best:
            return "win"
        elif my_score > opp_best:
            return "lose"
        return "draw"

    def remaining_cards(self, player_idx: int = 0) -> int:
        """
        学習側プレイヤーの残カード枚数を返す  
        *評価コールバック用*
        """
        return len(self.hands[player_idx])
