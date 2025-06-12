#!/usr/bin/env python3
"""
collect_and_finetune.py
────────────────────────────────────────────────────────
- Optuna で lr / ent_coef を探索 (300k step × trials)
- 中断再開可：既存 trial 数を確認し「不足分だけ」追加実行
- 観測は hand / board / current_player のみ
- player-1 は完全ランダム
"""

import patch_maskable

import argparse, warnings, random, time
from pathlib import Path
import numpy as np
import optuna

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

warnings.filterwarnings("ignore", category=UserWarning)

# ───────── settings ─────────
RAND_SEED = 42
random.seed(RAND_SEED); np.random.seed(RAND_SEED)

TRIAL_STEP     = 300_000
TRIAL_EVAL_EP  = 20
FINAL_STEP     = 1_000_000
FINAL_EVAL_EP  = 100

STORAGE    = "sqlite:///optuna_penguin.db"
STUDY_NAME = "penguin_lr_ent"
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL = MODELS_DIR / "optuna_best.zip"
# ────────────────────────────

# ---------- 観測フィルタ ----------
class ObsFilter(PenguinPartyGymEnv):
    """hand/board/current_player のみ返す"""
    def _get_observation(self):
        base = super()._get_observation()
        if base is None:
            return None
        return {"hand": base["hand"],
                "board": base["board"],
                "current_player": base["current_player"]}

# ---------- ランダム相手 ----------
class RandomOpponentWrapper(ObsFilter):
    @property
    def current_player(self): return getattr(self, "env", self).current_player
    def outcome(self, idx:int=0): return getattr(self, "env", self).outcome(idx)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        done = term or trunc
        while not done and self.current_player != 0:
            mask = self.action_masks()
            act = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, info = super().step(act)
            done = term or trunc
        return obs, rew, term, trunc, info

def env_fn():
    return ActionMasker(RandomOpponentWrapper(),
                        lambda e: e.action_masks())

# ---------- 評価 util ----------
def win_rate(model, n_ep=TRIAL_EVAL_EP):
    env = Monitor(RandomOpponentWrapper())
    wins = 0
    for _ in range(n_ep):
        obs, _ = env.reset(); done = False
        while not done:
            mask = env.unwrapped.action_masks()
            if env.unwrapped.current_player == 0:
                act, _ = model.predict(obs, deterministic=True,
                                       action_masks=mask)
            else:
                act = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        wins += env.unwrapped.outcome() == "win"
    return wins / n_ep * 100

# ---------- Optuna objective ----------
def objective(trial: optuna.Trial):
    lr  = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    ent = trial.suggest_float("ent",1e-4, 1e-2, log=True)
    env = make_vec_env(env_fn, n_envs=16)
    model = MaskablePPO("MultiInputPolicy", env,
                        learning_rate=lr, ent_coef=ent,
                        n_steps=2048, batch_size=512, verbose=0)
    model.learn(total_timesteps=TRIAL_STEP)
    wr = win_rate(model, TRIAL_EVAL_EP)
    trial.set_user_attr("win_rate", wr)
    return wr

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20, help="追加したい試行数")
    p.add_argument("--timeout",type=int, default=None, help="追加探索の秒制限")
    args = p.parse_args()

    study = optuna.create_study(direction="maximize",
                                study_name=STUDY_NAME,
                                storage=STORAGE,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())

    # 既に完了した trial 数
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.trials - completed)
    print(f"[Optuna] completed={completed}, requesting add={args.trials}, actually run={remaining}")

    if remaining > 0:
        study.optimize(objective,
                       n_trials=remaining,
                       timeout=args.timeout,
                       show_progress_bar=True)
    else:
        print("No additional trials needed; skipping optimization.")

    # ---- summary ----
    try:
        df = study.trials_dataframe()   # Optuna ≥2.5
    except AttributeError:
        df = study.trial_dataframe()    # 古い Optuna
    best = study.best_params
    print("\n==== Optuna Summary ====")
    print(df[["number","value","params_lr","params_ent"]]
          .sort_values("value",ascending=False).head())

    # ---- final training with best hyperparams ----
    print(f"\n[Final] lr={best['lr']:.3e}, ent={best['ent']:.3e}")
    env = make_vec_env(env_fn, n_envs=16)
    model = MaskablePPO("MultiInputPolicy", env,
                        learning_rate=best["lr"],
                        ent_coef=best["ent"],
                        n_steps=2048, batch_size=512, verbose=1)
    model.learn(total_timesteps=FINAL_STEP, progress_bar=True)

    wr = win_rate(model, FINAL_EVAL_EP)
    model.save(BEST_MODEL)
    print(f"\n[Final] win={wr:.1f}% | draw=--.-% | reward=--.-- | remain=--.--")  # 詳細は必要に応じて
    print("✅ saved →", BEST_MODEL)

if __name__ == "__main__":
    main()
