# ai.py  ────────────────────────────────────────────
# --------------------------------------------------
#  役割：
#    ・学習済み MaskablePPO モデルを読み込み
#    ・盤面・手札から合法手の確率分布を計算
#    ・最も確率の高い手を返す
#    ・全合法手と確率も併せて可視化用に返却
# --------------------------------------------------

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from sb3_contrib import MaskablePPO
from penguin_party_gym import PenguinPartyGymEnv

# ===== 定数 =====
DEFAULT_MODEL_PATH = Path("finetuning/ppo_penguin_league.zip")  # 既定のモデルパス
CANON_COLORS = ["Green", "Red", "Blue", "Yellow", "Purple"]     # 正規化カラー
LOWER2CANON = {c.lower(): c for c in CANON_COLORS}             # 小文字→正規表記

# ==================================================
#  エージェント本体
# ==================================================
class PPOAgent:
    """
    - `self.model` : Stable-Baselines3 の MaskablePPO
    - `self.stub`  : 行動マップ／マスク生成用のダミー環境
    """

    def __init__(self, model_path: str | Path):
        # モデルをロード
        self.model = MaskablePPO.load(str(model_path))
        # 行動マップ・マスク作成用ダミー環境（学習には使わない）
        self.stub = PenguinPartyGymEnv()
        print(f"[AI] model loaded → {model_path}")

    # ------------------------------------------------------------------
    #  内部ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _canon(c: str) -> str:
        """色名を大小無視で正規化 ("red" -> "Red")"""
        return LOWER2CANON.get(c.lower(), c.capitalize())

    def _fill_stub(self, board: Dict[str, str], hand: List[str]):
        """
        受け取った盤面／手札を stub 環境にコピーし、
        `action_masks()` が呼べる状態を作る。
        """
        env = self.stub.env
        env.board = {pos: self._canon(col) for pos, col in board.items()}
        env.hands = [[self._canon(c) for c in hand], []]  # プレイヤー1 だけ手札
        env.current_player = 0
        env.skipped.clear()
        return env

    def _encode_with_mask(self):
        """
        stub 環境から
          - NN 入力 obs (dict → tensor 変換前)
          - 合法手マスク (np.ndarray[bool])
        を取得して返す。
        """
        obs = self.stub._encode_obs({
            "hand": self.stub.env.hands[0],
            "board": self.stub.env.board,
            "current_player": 0,
        })
        mask = self.stub.action_masks()
        return obs, mask

    # ------------------------------------------------------------------
    #  公開メソッド
    # ------------------------------------------------------------------
    def get_action_distribution(
        self, board: Dict[str, str], hand: List[str]
    ) -> List[Tuple[str | None, str | None, float]]:
        """
        合法手ごとに (位置, カード, 確率) を返す。

        Parameters
        ----------
        board : {"row-col": "color", ...}
        hand  : ["red", "green", ...]

        Returns
        -------
        List[(pos, card, prob)]
        """
        # 1) stub 環境を現在状態に書き換え
        self._fill_stub(board, hand)
        obs, mask = self._encode_with_mask()

        # 2) obs → tensor 変換
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)

        # 3) マスク付きで行動分布 (Categorical) を取得
        dist = self.model.policy.get_distribution(
            obs_tensor, action_masks=torch.tensor(mask)[None]
        )
        # probs: shape=(アクション数,)
        probs: torch.Tensor = dist.distribution.probs.squeeze(0)

        # 4) numpy へ変換し (pos, card, prob) を作成
        probs_np = probs.detach().cpu().numpy()
        out: List[Tuple[str | None, str | None, float]] = []
        for idx, p in enumerate(probs_np):
            pos, card = self.stub.action_map[idx]
            if mask[idx]:                                   # 合法手のみ
                out.append((pos, card.lower() if card else card, float(p)))
        return out

    def select_action(
        self, board: Dict[str, str], hand: List[str]
    ) -> Tuple[str | None, str | None]:
        """
        盤面・手札から “最も確率の高い” 手を 1 つ返す。

        Returns
        -------
        (position | None, color | None)   ※ color は小文字
        """
        self._fill_stub(board, hand)
        obs, mask = self._encode_with_mask()
        # deterministic=True で argmax
        act_idx, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        pos, card = self.stub.action_map[int(act_idx)]
        return pos, card.lower() if card else card


# ==================================================
#  モジュール外部に公開するヘルパ関数
# ==================================================
_agent: PPOAgent | None = None

def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """
    モデルをロードしてグローバル _agent にセット。
    client 側から動的に呼び出すことも可能。
    """
    global _agent
    _agent = PPOAgent(path)

# モジュール import 時にデフォルトモデルをロード
load_model(DEFAULT_MODEL_PATH)


def Evaluation(hand: List[str] | None = None):
    """
    すべての色を 0.20 固定で返す
    （サーバは合計 1.00 でなくても受理する仕様のため OK)
    """
    return [[c, 0.20] for c in CANON_COLORS]
    
    """
    サーバの “色評価値” 要求に応答する関数。
    現在は「手札に占める割合」を返す簡易版。
    """
    if not hand:
        return [[c, 0.50] for c in CANON_COLORS]  # 手札不明ならフラット
    total = len(hand)
    hand_low = [h.lower() for h in hand]
    vals = {c: round(hand_low.count(c.lower()) / total, 2) for c in CANON_COLORS}
    return [[c, vals[c]] for c in CANON_COLORS]


def ModelPlay(board: Dict[str, str], hand: List[str]):
    """
    1 手選択＋全合法手分布を返す統合インターフェース。

    Returns
    -------
    pos  : str | None  … 置く位置 ("1-7" 等) / None=スキップ
    card : str | None  … 置く色 (小文字) / None=スキップ
    dist : List[(pos, card, prob)]  … 全合法手と確率
    """
    if _agent is None:
        load_model()

    dist = _agent.get_action_distribution(board, hand)
    pos, card = _agent.select_action(board, hand)
    return pos, card, dist
