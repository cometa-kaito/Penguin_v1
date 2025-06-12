#!/usr/bin/env python3
"""
finetune_and_search.py
────────────────────────────────────────────────────────
player-0 : 追加学習する PPO
player-1 : 凍結モデル（過去ベスト）
観測     : hand / board / current_player (+action_mask)
評価     : **モデル vs ランダム AI** で win/draw を測定
"""

import patch_maskable  # robust renormalization patch

from pathlib import Path
import numpy as np
import torch
from torch import optim

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

# ───────── Hyperparams ─────────
TOTAL_STEPS = 500_000
N_ENVS      = 8
N_STEPS     = 4096
POLICY_LR, VALUE_LR = 1e-4, 3e-4
ENT_COEF, VF_COEF, GAE_LAMBDA = 2.9e-4, 1.0, 0.9
FIXED_MODEL_PATH = "models/optuna_finetuned.zip"   # player-1
SAVE_FINAL = Path("models/league_final.zip")
# ───────────────────────────────

# ╭───────────────────────────╮
# │     観測フィルタ          │
# ╰───────────────────────────╯
class ObsFilter(PenguinPartyGymEnv):
    def _get_observation(self):
        base = super()._get_observation()
        if base is None:
            return None
        return {"hand": base["hand"],
                "board": base["board"],
                "current_player": base["current_player"]}

# ╭───────────────────────────╮
# │  Self-play 環境 (固定 vs 学習) │
# ╰───────────────────────────╯
fixed_policy = MaskablePPO.load(FIXED_MODEL_PATH).policy

class SelfPlayWrapper(ObsFilter):
    @property
    def current_player(self):
        return getattr(self, "env", self).current_player
    def outcome(self, idx=0):
        return getattr(self, "env", self).outcome(idx)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        done = term or trunc
        while not done and self.current_player != 0:
            mask = self.action_masks()
            act, _ = fixed_policy.predict(obs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, info = super().step(act)
            done = term or trunc
        return obs, rew, term, trunc, info

# ╭───────────────────────────╮
# │  Random 対戦環境 (評価用) │
# ╰───────────────────────────╯
class RandomOpponentWrapper(ObsFilter):
    @property
    def current_player(self):
        return getattr(self, "env", self).current_player
    def outcome(self, idx=0):
        return getattr(self, "env", self).outcome(idx)
    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        done = term or trunc
        while not done and self.current_player != 0:
            mask = self.action_masks()
            rand_action = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, info = super().step(rand_action)
            done = term or trunc
        return obs, rew, term, trunc, info
    def remaining_cards(self, idx: int | None = None):
        """
        idx が None の場合はエージェント側 (player 0) を
        デフォルトとして扱う。
        """
        if idx is None:
            idx = 0
        base = getattr(self, "env", self)
        return base.remaining_cards(idx)

# ╭───────────────────────────╮
# │   Dual-LR Policy          │
# ╰───────────────────────────╯
class DualLRPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *a, lr_policy=POLICY_LR, lr_value=VALUE_LR, **kw):
        self.lr_p, self.lr_v = lr_policy, lr_value
        super().__init__(*a, **kw)
    def _make_optimizer(self):
        pi, vf = [], []
        for n, p in self.named_parameters():
            (vf if "value_net" in n else pi).append(p)
        self.optimizer = optim.Adam(
            [{"params": pi, "lr": self.lr_p},
             {"params": vf, "lr": self.lr_v}],
            **self.optimizer_kwargs
        )

# ╭───────────────────────────╮
# │        評価 util          │
# ╰───────────────────────────╯
def evaluate(model, n_ep=100):
    env = Monitor(RandomOpponentWrapper())
    wins = draws = tot_r = rem = 0
    for ep in range(n_ep):
        obs, _ = env.reset(); done = False
        while not done:
            mask = env.unwrapped.action_masks()
            if env.unwrapped.current_player == 0:
                act, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                act = int(np.random.choice(np.flatnonzero(mask)))
            obs, r, term, trunc, _ = env.step(act)
            tot_r += r; done = term or trunc
        res = env.unwrapped.outcome()
        wins += res == "win"; draws += res == "draw"
        rem  += env.unwrapped.remaining_cards()
    return dict(win=wins/n_ep*100,
                draw=draws/n_ep*100,
                reward=tot_r/n_ep,
                remain=rem/n_ep)

# ╭───────────────────────────╮
# │          main            │
# ╰───────────────────────────╯
def main():
    print("[Start] Self-play PPO (player-1 fixed)")

    vec_env = make_vec_env(
        lambda: ActionMasker(SelfPlayWrapper(),
                             lambda e: e.action_masks()),
        n_envs=N_ENVS
    )

    model = MaskablePPO(DualLRPolicy, vec_env,
                        learning_rate=POLICY_LR, ent_coef=ENT_COEF,
                        vf_coef=VF_COEF, n_steps=N_STEPS,
                        batch_size=512, gae_lambda=GAE_LAMBDA,
                        verbose=1)

    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    fin = evaluate(model, 100)

    print(f"[Final] win={fin['win']:.1f}% | draw={fin['draw']:.1f}% "
          f"| reward={fin['reward']:.2f} | remain={fin['remain']:.2f}")

    model.save(SAVE_FINAL)
    print("✓ Saved →", SAVE_FINAL.resolve())

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
