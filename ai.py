# ai.py ────────────────────────────────────────────
# --------------------------------------------------
#  役割：
#    ・学習済み MaskablePPO モデルを読み込み
#    ・盤面・手札から合法手の確率分布を計算
#    ・最も確率の高い手を返す
# --------------------------------------------------

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from penguin_party_gym import PenguinPartyGymEnv

# ===== 定数 =====
DEFAULT_MODEL_PATH = Path("models/critic_tuned_v2.zip")
CANON_COLORS = ["Green", "Red", "Blue", "Yellow", "Purple"]
LOWER2CANON = {c.lower(): c for c in CANON_COLORS}

# ==================================================
#  エージェント本体
# ==================================================
class PPOAgent:
    def __init__(self, model_path: str | Path):
        self.model = MaskablePPO.load(str(model_path))
        self.stub = PenguinPartyGymEnv()  # マスク生成用ダミー環境
        print(f"[AI] model loaded → {model_path}")

    # ---------- 内部ユーティリティ ----------
    @staticmethod
    def _canon(c: str) -> str:
        return LOWER2CANON.get(c.lower(), c.capitalize())

    def _fill_stub(self, board: Dict[str, str], hand: List[str]):
        env = self.stub.env
        env.board = {p: self._canon(col) for p, col in board.items()}
        env.hands = [[self._canon(c) for c in hand], []]
        env.current_player = 0
        env.skipped.clear()
        return env

    def _encode_with_mask(self):
        """stub 環境から (obs, mask) を生成して返す。"""
        obs_base = self.stub._encode_obs({
            "hand": self.stub.env.hands[0],
            "board": self.stub.env.board,
            "current_player": 0,
        })
        mask = self.stub.action_masks().astype(np.int8)

        # dict と ndarray の両対応
        if isinstance(obs_base, dict):
            obs = obs_base.copy()
            obs["action_mask"] = mask
        else:
            obs = {
                "observation": obs_base.astype(np.float32),
                "action_mask": mask
            }
        return obs, mask

    # ---------- 公開メソッド ----------
    def get_action_distribution(
        self, board: Dict[str, str], hand: List[str]
    ) -> List[Tuple[str | None, str | None, float]]:
        """合法手ごとに (位置, カード, 確率) を返す。"""
        self._fill_stub(board, hand)
        obs, mask = self._encode_with_mask()

        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        dist = self.model.policy.get_distribution(
            obs_tensor, action_masks=torch.tensor(mask)[None]
        )
        probs_np = dist.distribution.probs.squeeze(0).detach().cpu().numpy()

        out: List[Tuple[str | None, str | None, float]] = []
        for idx, p in enumerate(probs_np):
            if mask[idx] == 0:
                continue  # 非合法手は除外
            pos, card = self.stub.action_map[idx]
            out.append((pos, card.lower() if card else card, float(p)))
        return out

    def select_action(
        self, board: Dict[str, str], hand: List[str]
    ) -> Tuple[str | None, str | None]:
        """盤面・手札から最も確率の高い手を 1 つ返す。"""
        self._fill_stub(board, hand)
        obs, mask = self._encode_with_mask()
        act_idx, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        pos, card = self.stub.action_map[int(act_idx)]
        return pos, card.lower() if card else card


# ==================================================
#  モジュール外部に公開するヘルパ関数
# ==================================================
_agent: Optional[PPOAgent] = None

def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    global _agent
    _agent = PPOAgent(path)

# import 時にデフォルトモデルをロード
load_model(DEFAULT_MODEL_PATH)


def Evaluation(hand: List[str] | None = None):
    """サーバからの色評価要求に返すダミー実装。"""
    return [[c, 0.20] for c in CANON_COLORS]


def ModelPlay(board: Dict[str, str], hand: List[str]):
    """1 手選択＋全合法手分布を返す統合インターフェース。"""
    if _agent is None:
        load_model()
    dist = _agent.get_action_distribution(board, hand)
    pos, card = _agent.select_action(board, hand)
    return pos, card, dist
