#!/usr/bin/env python3
"""
collect_and_finetune.py
────────────────────────────────────────────────────────
  1. Optuna で lr / ent_coef を粗探索 (300k step × trial)
     - 既存 study があれば不足分だけ追加実行
  2. 既に存在する models/best/best_model.zip を読み込み
     FINAL_STEP 追加学習し，結果を表示
観測: hand / board / current_player (+ action_mask)
相手: ランダム
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

# ───── 固定設定 ─────
RAND_SEED = 42
random.seed(RAND_SEED); np.random.seed(RAND_SEED)

TRIAL_STEP     = 300_000
TRIAL_EVAL_EP  = 20
FINAL_STEP     = 700_000
FINAL_EVAL_EP  = 100

STORAGE    = "sqlite:///optuna_penguin.db"
STUDY_NAME = "penguin_lr_ent"
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL = MODELS_DIR / "best" / "best_model.zip"   # ← 追加学習のベース
FINAL_OUT  = MODELS_DIR / "optuna_finetuned.zip"
# ───────────────────

# ───── 環境ラッパ (自手札+盤面のみ) ─────
class ObsFilter(PenguinPartyGymEnv):
    def _get_observation(self):
        base = super()._get_observation()
        if base is None: return None
        return {"hand": base["hand"], "board": base["board"],
                "current_player": base["current_player"]}

class RandomOpponentWrapper(ObsFilter):
    @property
    def current_player(self): return getattr(self,"env",self).current_player
    def outcome(self, idx:int=0): return getattr(self,"env",self).outcome(idx)
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        while not (term or trunc) and self.current_player != 0:
            mask = self.action_masks()
            rand_act = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, info = super().step(rand_act)
        return obs, r, term, trunc, info

def env_fn():
    return ActionMasker(RandomOpponentWrapper(),
                        lambda e: e.action_masks())

# ───── 評価 util ─────
def win_rate(model, n_ep=TRIAL_EVAL_EP):
    env = Monitor(RandomOpponentWrapper())
    wins = 0
    for _ in range(n_ep):
        obs,_ = env.reset(); done=False
        while not done:
            mask = env.unwrapped.action_masks()
            if env.unwrapped.current_player==0:
                act,_ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                act = int(np.random.choice(np.flatnonzero(mask)))
            obs,_,term,trunc,_ = env.step(act)
            done = term or trunc
        wins += env.unwrapped.outcome()=="win"
    return wins/n_ep*100

# ───── Optuna objective ─────
def objective(trial):
    lr  = trial.suggest_float("lr",1e-5,5e-4,log=True)
    ent = trial.suggest_float("ent",1e-4,1e-2,log=True)
    env = make_vec_env(env_fn, n_envs=16)
    model = MaskablePPO("MultiInputPolicy", env,
                        learning_rate=lr, ent_coef=ent,
                        n_steps=2048, batch_size=512, verbose=0)
    model.learn(total_timesteps=TRIAL_STEP)
    wr = win_rate(model, TRIAL_EVAL_EP)
    trial.set_user_attr("win_rate", wr)
    return wr

# ───── main ─────
def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--trials", type=int, default=20)
    arg.add_argument("--timeout", type=int, default=None)
    args = arg.parse_args()

    study = optuna.create_study(direction="maximize",
                                study_name=STUDY_NAME,
                                storage=STORAGE,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())

    done_trials = len([t for t in study.trials
                       if t.state==optuna.trial.TrialState.COMPLETE])
    remain = max(0, args.trials - done_trials)
    print(f"[Optuna] completed={done_trials}, running extra={remain}")
    if remain:
        study.optimize(objective, n_trials=remain,
                       timeout=args.timeout, show_progress_bar=True)

    # ------ summary ------
    try:
        df = study.trials_dataframe()
    except AttributeError:
        df = study.trial_dataframe()
    print("\n==== Optuna Best 5 ====")
    print(df.sort_values("value", ascending=False)
            [["number","value","params_lr","params_ent"]].head())

    best = study.best_params
    print(f"\n[Finetune] continue from {BEST_MODEL}")
    env = make_vec_env(env_fn, n_envs=16)

    if BEST_MODEL.exists():
        model = MaskablePPO.load(BEST_MODEL, env=env)
        print("✓ model loaded.")
    else:
        # fallback: 新規モデルを作成
        model = MaskablePPO("MultiInputPolicy", env,
                            learning_rate=best["lr"], ent_coef=best["ent"],
                            n_steps=2048, batch_size=512, verbose=1)
        print("! best_model.zip not found → 新規作成")

    model.learn(total_timesteps=FINAL_STEP, progress_bar=True)
    wr = win_rate(model, FINAL_EVAL_EP)
    model.save(FINAL_OUT)

    print(f"\n[Final] win={wr:.1f}% | draw=--.-% | reward=--.-- | remain=--.--")
    print("✅ finetuned model →", FINAL_OUT)

if __name__ == "__main__":
    main()
