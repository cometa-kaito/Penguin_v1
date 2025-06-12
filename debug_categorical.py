import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical

def patched_init(self, logits=None, **kwargs):
    super(MaskableCategorical, self).__init__(logits=logits, **kwargs)
    with torch.no_grad():
        probs = self.probs.detach().clone()   # shape: (batch, n_act)
        row_sum = probs.sum(-1)
        bad = (row_sum - 1).abs() > 1e-3
        if bad.any():
            i = torch.where(bad)[0][0]        # 最初の壊れ行
            torch.save({
                "logits": logits[i].cpu(),
                "probs" : probs[i].cpu(),
                "row_sum": row_sum[i].item()
            }, "bad_distribution.pt")
            print(f"[DBG] simplex broke: row_sum={row_sum[i]:.3f}")
            raise RuntimeError("Simplex-violation captured")
MaskableCategorical.__init__ = patched_init  # monkey-patch
