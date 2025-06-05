import random
from typing import Dict, List, Tuple, Optional

class PenguinPartyEnv:
    """
    2 人対戦・手札 14 枚版ペンギンパーティー環境クラス
    --------------------------------------------------
    board のキー   : "row-col" 例 "3-7"
    board の value : カラー文字列 ("Green" など)
    """

    def __init__(
        self,
        num_players: int = 2,
        max_hand_size: int = 14,
        colors: Optional[List[str]] = None,
        deck_distribution: Optional[Dict[str, int]] = None,
        reward_config: Optional[Dict[str, float]] = None,
        initial_position_policy: str = "center-only",  # または "free"
    ):
        # ----------- 基本設定 -----------
        self.num_players    = num_players          # プレイヤー人数
        self.max_hand_size  = max_hand_size        # 初期手札枚数/人
        self.colors         = colors or ["Green", "Red", "Blue", "Yellow", "Purple"]

        # ----------- デッキ構成 -----------
        # カラーごとの枚数を dict で指定。デフォルトは合計 36 枚。
        self.deck_distribution = deck_distribution or {
            "Green": 8,
            "Red": 7,
            "Blue": 7,
            "Yellow": 7,
            "Purple": 7,
        }

        # ----------- 報酬テーブル -----------
        # 強化学習用：最小限のモチベーション設計
        self.reward_config = reward_config or {
            "valid_action_bonus":     0.05,  # 合法手を置くごとに微報酬
            "win_base":               1.0,   # 勝利時の基本報酬
            "win_per_diff":           0.1,   # 勝敗差 (枚数差) に比例
            "lose_base":             -1.0,   # 敗北時の基本罰
            "lose_per_diff":          0.1,   # 敗北時も差分で追加罰
            "invalid_action_penalty":-0.2,   # 無効手 (強制スキップ) 罰
        }

        # 最下段 1 手目の配置ルール
        assert initial_position_policy in ("center-only", "free")
        self.initial_position_policy = initial_position_policy

        # 初期化
        self.reset()

    # ------------------------------------------------------------------ #
    #  基本メソッド
    # ------------------------------------------------------------------ #
    def reset(self):
        """山札生成 → シャッフル → 各プレイヤーに配布し盤面を初期化"""
        deck: List[str] = []
        for color, count in self.deck_distribution.items():
            deck.extend([color] * count)
        random.shuffle(deck)

        # 各プレイヤーの手札
        self.hands: List[List[str]] = [
            [deck.pop() for _ in range(self.max_hand_size)]
            for _ in range(self.num_players)
        ]

        self.board: Dict[str, str] = {}      # 盤面 (座標 → カラー)
        self.current_player: int = 0         # 先手は P0
        self.done: bool = False
        self.skipped: set[int] = set()       # スキップ宣言済プレイヤー集合
        return self._get_observation()

    def _get_observation(self):
        """観測オブジェクトを返す (現在手番のみ)"""
        if self.done or self.current_player is None:
            return None
        return {
            "hand": list(self.hands[self.current_player]),  # 手札コピー
            "board": dict(self.board),                     # 盤面コピー
            "current_player": self.current_player,
        }

    # ------------------------------------------------------------------ #
    #  ゲーム進行 (行動 = (position, color))
    # ------------------------------------------------------------------ #
    def step(self, action: Tuple[Optional[str], Optional[str]]):
        """
        Parameters
        ----------
        action : (position, color)
          - 通常置き手      … ("3-7", "Red") のように座標と色を指定
          - スキップ        … (None, None)
        Returns
        -------
        obs, reward, done, info
        """
        # ───────── 終了後に step した場合 ─────────
        if self.done or self.current_player is None:
            return self._get_observation(), 0.0, True, {"info": "game done"}

        position, color = action
        valid_actions = self.get_valid_actions()

        # ---------- 無効手 ----------
        if (position, color) not in valid_actions:
            self.skipped.add(self.current_player)
            next_player = self._next_active_player()
            self.current_player = next_player
            if next_player is None:          # 全員スキップ済なら終了
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
                # 全員スキップで即終了
                self.done = True
                scores = self.get_scores()
                reward = self._calc_end_reward(self.current_player, scores)
                return None, reward, self.done, {"info": "all skipped (game over)"}

            # ゲーム継続：次のアクティブプレイヤーへ
            self.current_player = self._next_active_player()
            return self._get_observation(), 0.0, self.done, {"info": "skip"}

        # ---------- 正常手 ----------
        self.board[position] = color
        self.hands[self.current_player].remove(color)

        # 盤面 or 手札が空になったらゲーム終了フラグ
        if all(len(h) == 0 for h in self.hands):
            self.done = True

        next_player = self._next_active_player()
        if next_player is None:
            self.done = True

        # 報酬：合法手ボーナス +（終了時のみ）勝敗報酬
        reward = self.reward_config["valid_action_bonus"]
        if self.done:
            scores = self.get_scores()
            reward += self._calc_end_reward(self.current_player, scores)

        # 次手番へ（終了時はそのまま）
        self.current_player = next_player if not self.done else self.current_player
        return self._get_observation(), reward, self.done, {}

    # ------------------------------------------------------------------ #
    #  ターン遷移ヘルパ
    # ------------------------------------------------------------------ #
    def _next_active_player(self) -> Optional[int]:
        """
        現在の次にスキップしていないプレイヤーを時計回りで探す。
        全員スキップなら None。
        """
        start = self.current_player
        for i in range(1, self.num_players + 1):
            cand = (start + i) % self.num_players
            if cand not in self.skipped:
                return cand
        return None  # 全員スキップ

    # ------------------------------------------------------------------ #
    #  ルール判定
    # ------------------------------------------------------------------ #
    def is_valid_action(self, position: str, color: str) -> bool:
        """
        与えられた座標・色が合法手かどうかを判定する。
        * 盤面空き & 手札保有チェック
        * 座標フォーマットおよび範囲
        * ピラミッド構造の上下関係ルール
        """
        # 既に埋まっている／手札に無い
        if position in self.board or color not in self.hands[self.current_player]:
            return False
        try:
            row, col = map(int, position.split("-"))
        except ValueError:
            return False

        # 座標範囲：row 1–7, col 1–13
        if not (1 <= row <= 7) or not (1 <= col <= 13):
            return False

        # --- 最下段判定 ---
        if row == 1:
            # 既に置かれている最下段列を取得
            base_cols = sorted(int(k.split("-")[1]) for k in self.board if k.startswith("1-"))
            if not base_cols:
                # 1 手目は中央 (col=7) or 自由 (設定による)
                return col == 7 if self.initial_position_policy == "center-only" else True
            # 最下段は最大 7 枚 → それ以上は置けない
            if len(base_cols) >= 7:
                return False
            # 連続で並ぶよう端に追加
            return col == base_cols[0] - 1 or col == base_cols[-1] + 1

        # --- 上段判定 ---
        left  = f"{row-1}-{col}"
        right = f"{row-1}-{col+1}"
        # 下段に 2 枚置かれていなければ不可
        if left not in self.board or right not in self.board:
            return False
        # 下段いずれかと同色なら OK
        return color in (self.board[left], self.board[right])

    # --- 合法手集合を取得 ---
    def get_valid_actions(self):
        return self.get_valid_actions_for(self.current_player)

    def get_valid_actions_for(self, player_idx: int):
        if player_idx is None or player_idx >= self.num_players:
            return [(None, None)]  # 異常時はスキップのみ
        valid: List[Tuple[Optional[str], Optional[str]]] = []
        hand = self.hands[player_idx]
        for row in range(1, 8):          # row 1–7
            for col in range(1, 14):     # col 1–13
                pos = f"{row}-{col}"
                for c in set(hand):      # 同色はまとめて判定
                    if self.is_valid_action(pos, c):
                        valid.append((pos, c))
        if not valid:
            valid.append((None, None))   # 合法手無し → スキップ
        return valid

    # ------------------------------------------------------------------ #
    #  スコア・報酬
    # ------------------------------------------------------------------ #
    def get_scores(self):
        """各プレイヤーの残り手札枚数リストを返す"""
        return [len(h) for h in self.hands]

    def get_winner(self) -> int:
        """
        勝者インデックス (0 or 1) を返す。引き分け時は -1。
        """
        scores = self.get_scores()
        min_s  = min(scores)
        winners = [i for i, s in enumerate(scores) if s == min_s]
        return winners[0] if len(winners) == 1 else -1

    def _calc_end_reward(self, current_player: int, scores: List[int]) -> float:
        """
        ゲーム終了時の追加報酬を計算。
        自分と最良対戦相手の枚数差で win/lose 報酬を計算する。
        """
        my_score   = scores[current_player]
        opp_scores = [s for i, s in enumerate(scores) if i != current_player]
        if not opp_scores:            # 1 人プレイは想定外
            return 0.0
        opp_min = min(opp_scores)
        diff    = opp_min - my_score
        if my_score < opp_min:        # 勝利
            return self.reward_config["win_base"] + self.reward_config["win_per_diff"] * diff
        elif my_score > opp_min:      # 敗北
            return self.reward_config["lose_base"] + self.reward_config["lose_per_diff"] * diff
        else:                         # 引き分け
            return 0.0

    # ------------------------------------------------------------------ #
    #  描画（デバッグ用）
    # ------------------------------------------------------------------ #
    def render(self):
        """テキストで盤面と手札を表示（学習デバッグ確認用）"""
        print("\n=== Board (Top → Bottom) ===")
        row_to_cols = {
            7: range(1, 8),   6: range(1, 9),  5: range(1, 10),
            4: range(1, 11),  3: range(1, 12), 2: range(1, 13),
            1: range(1, 14),
        }
        width = 13 * 6  # 行幅 (センタリング用)
        for row in range(7, 0, -1):
            line = "".join(
                f"{self.board.get(f'{row}-{col}', '___')[:3]:^6}"
                for col in row_to_cols[row]
            )
            print(line.center(width))

        print("\n=== Hands ===")
        for i, h in enumerate(self.hands):
            print(f"P{i}: {h} ({len(h)}枚)")
        print(f"Current Player: {self.current_player}")
        print("====================\n")
