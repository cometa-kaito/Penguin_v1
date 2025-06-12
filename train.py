#!/usr/bin/env python3
"""
train.py – Maskable-PPO (player0 = 学習, player1 = ランダム)
観測は【自分の手札・盤面のみ】に制限。
"""

import patch_maskable                                         # Simplex パッチ

from pathlib import Path
import numpy as np
import torch
from torch import optim

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from penguin_party_gym import PenguinPartyGymEnv

# ─── Hyperparams ───
POLICY_LR, VALUE_LR = 1e-4, 3e-4
TOTAL_STEP, N_ENVS, N_STEPS = 2_000_000, 16, 4096
VF_COEF, GAE_LAMBDA, ENT_COEF = 1.0, 0.9, 2.905e-4
EVAL_EPISODES, EVAL_FREQ = 100, 80_000
SAVE_FINAL = Path("models/critic_tuned_v2.zip")
SAVE_BEST_DIR = Path("models/best")
SAVE_FINAL.parent.mkdir(exist_ok=True, parents=True)
SAVE_BEST_DIR.mkdir(exist_ok=True, parents=True)
# ───────────────────


# ───────── 観測フィルタ ─────────
class ObsFilterWrapper(PenguinPartyGymEnv):
    """
    observation を {'hand': ..., 'board': ..., 'current_player': ...}
    だけに縮小。相手手札情報や残カード数は一切渡さない。
    """

    def _get_observation(self):
        base = super()._get_observation()
        if base is None:
            return None
        # hand は player0 のみ / board はそのまま
        return {
            "hand": base["hand"],
            "board": base["board"],
            "current_player": base["current_player"],
        }
# ────────────────────────────────


# ─── Random-opponent wrapper ───
class RandomOpponentWrapper(ObsFilterWrapper):
    """
    player1 = ランダム一様抽出
    player0 の報酬だけ返す
    """

    # delegate helper
    @property
    def current_player(self):
        return self.env.current_player

    def outcome(self, idx: int = 0):
        return self.env.outcome(idx)

    def remaining_cards(self, idx: int = 0):
        return self.env.remaining_cards(idx)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        done = term or trunc

        def add_final(r, inf):
            if done and "final_rewards" not in inf:
                fr = self._terminal_rewards()
                r += fr[0]
                inf["final_rewards"] = fr
            return r, inf

        while not done and self.current_player != 0:
            mask = self.action_masks()
            rand_act = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, info = super().step(rand_act)
            done = term or trunc

        rew, info = add_final(rew, info)
        return obs, rew, term, trunc, info
# ─────────────────────────────────


class DualLRPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *a, lr_policy=POLICY_LR, lr_value=VALUE_LR, **kw):
        self.lr_p, self.lr_v = lr_policy, lr_value
        super().__init__(*a, **kw)

    def _make_optimizer(self):
        pi, vf = [], []
        for n, p in self.named_parameters():
            (vf if "value_net" in n else pi).append(p)
        self.optimizer = optim.Adam(
            [{"params": pi, "lr": self.lr_p}, {"params": vf, "lr": self.lr_v}],
            **self.optimizer_kwargs,
        )

# ─── 評価 util ───
def simple_eval(model, n_ep=EVAL_EPISODES):
    env = Monitor(RandomOpponentWrapper())
    rng = np.random.default_rng()
    w = d = r_sum = rem_sum = 0
    for ep in range(n_ep):
        obs, _ = env.reset()
        done = False
        while not done:
            if env.unwrapped.current_player == 0:
                mask = env.unwrapped.action_masks()
                act, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                valid = env.unwrapped.get_valid_actions()
                act = valid[rng.integers(len(valid))]
            obs, r, term, trunc, _ = env.step(act)
            r_sum += r
            done = term or trunc
        out = env.unwrapped.outcome()
        w += out == "win"; d += out == "draw"
        rem_sum += env.unwrapped.remaining_cards()
    return dict(win_rate=w/n_ep*100, draw_rate=d/n_ep*100,
                mean_reward=r_sum/n_ep, mean_remain=rem_sum/n_ep)

class EvalCB(BaseCallback):
    def __init__(self, freq, verbose=0):
        super().__init__(verbose); self.freq=freq; self.best=-np.inf
    def _on_step(self):
        if self.n_calls % self.freq == 0:
            st = simple_eval(self.model)
            self.logger.record("eval/win_rate", st["win_rate"])
            if st["win_rate"] > self.best:
                self.best = st["win_rate"]
                p = SAVE_BEST_DIR / "best_model"
                self.model.save(p)
                if self.verbose: print(f"[Eval] new best {self.best:.1f}% → {p}")
        return True

# ─── main ───
def main():
    print("[Start] PPO vs Random (own info only)")
    vec_env = make_vec_env(
        lambda: ActionMasker(RandomOpponentWrapper(), lambda e: e.action_masks()),
        n_envs=N_ENVS
    )
    model = MaskablePPO(
        DualLRPolicy, vec_env,
        learning_rate=POLICY_LR, ent_coef=ENT_COEF, vf_coef=VF_COEF,
        n_steps=N_STEPS, batch_size=512, gae_lambda=GAE_LAMBDA,
        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
        verbose=1, tensorboard_log="tb_logs",
    )
    model.learn(total_timesteps=TOTAL_STEP,
                callback=EvalCB(EVAL_FREQ,1), progress_bar=True)
    fin = simple_eval(model)
    print(f"\n[Final] win={fin['win_rate']:.1f}% | draw={fin['draw_rate']:.1f}% | "
          f"reward={fin['mean_reward']:.2f} | remain={fin['mean_remain']:.2f}")
    model.save(SAVE_FINAL)
    print("✓ Saved →", SAVE_FINAL.resolve())

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); main()
