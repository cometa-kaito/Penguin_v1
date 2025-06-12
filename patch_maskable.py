"""
patch_maskable.py  –  MaskableCategorical.apply_masking を安全に上書き
SB3 v2.6.* 以降対応版
------------------------------------------------------------
ポイント
1. 元実装 (_orig) は「マスク後 logits」を返すだけなので、
   それを受け取って **logits 側** を再正規化する。
2. softmax(logits) が厳密に 1.0 になるよう
   logits ← logits - logsumexp(logits) を行う。
3. 行頭の -inf や nan を処理して勾配崩壊を回避。
"""

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical

# 元の apply_masking を退避
_orig = MaskableCategorical.apply_masking

def _renormalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """各行 softmax が 1 になるよう logits をシフト"""
    # 万一 nan が紛れ込んでいた場合は -1e9 に置換
    logits = torch.where(torch.isnan(logits), torch.full_like(logits, -1e9), logits)
    # -inf だけの行はそのまま（後段が自動対応）
    with torch.no_grad():
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    logits = logits - logsumexp
    return logits

def patched(self, masks):
    """
    1. 既存実装でマスク適用
    2. logits を logsumexp 正規化
    3. return logits （None のときはそのまま）
    """
    logits = _orig(self, masks)              # -> logits or None
    if logits is None:                       # (__init__ の最初期呼び出し等)
        return None

    logits = _renormalize_logits(logits)

    # 既に distribution が存在する場合は手動で更新しておく
    if hasattr(self, "distribution"):
        with torch.no_grad():
            self.distribution.logits = logits
            self.distribution.probs  = torch.softmax(logits, dim=-1)

    return logits

# Monkey-patch
MaskableCategorical.apply_masking = patched
print("[patch_maskable] Safe logits-renormalization patch active")
