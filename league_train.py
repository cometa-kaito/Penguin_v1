#!/usr/bin/env python3
"""
league_train.py – Penguin Party 自己対戦リーグ（本学習ビルド, Apple-Silicon/MPS 最適化）
================================================================================================
フェーズ概要
-----------
phase-0 : Optuna 探索 (lr, ent_coef, …)  –  300 k step × n_trials
          └→ ベスト設定を 2 M step 追加学習
phase-1 : ランダム相手で 160 k step 追加学習
phase-2~: 直前までの自己プール相手で 160 k step ずつ回す（デフォルト 3 フェーズ）

実行環境
--------
* Apple M1/M2 (8 コア) を想定。  
* SB3 rollout は CPU マルチプロセス、NN forward/backward は MPS 自動利用。
* Phase-２以降は DummyVecEnv (単一プロセス) を使い pickle 問題を回避。

コマンド例
----------
# Optuna 40 trial, タイムアウト 6 時間、リーグ 3 phase
python league_train.py --trials 40 --timeout 21600 --phases 3
"""

from __future__ import annotations
import argparse, warnings, random, functools
from pathlib import Path
from typing import Optional, List

import numpy as np
import optuna
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from penguin_party_gym import PenguinPartyGymEnv

warnings.filterwarnings("ignore", category=UserWarning)

# ───────── RNG & DEVICE ─────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ───────── 本学習定数 ─────────
CPU_CORES, N_ENVS      = 8, 8         # M2 8-core
TRIAL_STEP, TRIAL_EP   = 300_000, 20
FINAL_STEP             = 2_000_000
PHASE_STEP             = 160_000

ROOT       = Path(__file__).parent.resolve()
SAVE_DIR   = ROOT / "league"; SAVE_DIR.mkdir(exist_ok=True)
OPTUNA_DB  = f"sqlite:///{ROOT / 'optuna_penguin.db'}"
STUDY_NAME = "penguin_lr_ent"
BEST_MODEL = SAVE_DIR / "optuna_best.zip"

# ───────── Env ラッパー ─────────
class ObsFilter(PenguinPartyGymEnv):
    def _get_observation(self):
        b = super()._get_observation()
        return None if b is None else {k: b[k] for k in ("hand", "board", "current_player")}

class RandomOpponent(ObsFilter):
    @property
    def current_player(self): return getattr(self, "env", self).current_player
    def outcome(self, idx=0):  return getattr(self, "env", self).outcome(idx)
    def step(self, act):
        obs, r, term, trunc, info = super().step(act); done = term or trunc
        while not done and self.current_player != 0:
            mask = self.action_masks()
            rand = int(np.random.choice(np.flatnonzero(mask)))
            obs, _, term, trunc, info = super().step(rand); done = term or trunc
        return obs, r, term, trunc, info

def env_fn_random():
    return ActionMasker(RandomOpponent(), lambda e: e.action_masks())

class FixedOpponent(RandomOpponent):
    def __init__(self, path: Path):
        super().__init__(); self._opp = MaskablePPO.load(path, device=DEVICE)
    def step(self, act):
        obs, r, term, trunc, info = super().step(act); done = term or trunc
        while not done and self.current_player != 0:
            mask = self.action_masks()
            a, _ = self._opp.predict(obs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, info = super().step(int(a)); done = term or trunc
        return obs, r, term, trunc, info

# ───────── Utils ─────────
@torch.no_grad()
def win_rate_vs_random(model: MaskablePPO, n_ep=TRIAL_EP):
    env, wins = Monitor(RandomOpponent()), 0
    for _ in range(n_ep):
        obs,_ = env.reset(); done=False
        while not done:
            mask = env.unwrapped.action_masks()
            a = model.predict(obs, deterministic=True, action_masks=mask)[0] \
                if env.unwrapped.current_player==0 \
                else int(np.random.choice(np.flatnonzero(mask)))
            obs,_,term,trunc,_ = env.step(a); done = term or trunc
        wins += env.unwrapped.outcome() == "win"
    return wins / n_ep * 100

def make_env(factory, multiproc: bool = True):
    vec_cls = SubprocVecEnv if multiproc else DummyVecEnv
    kwargs  = {"vec_env_cls": vec_cls}
    if multiproc:
        kwargs["vec_env_kwargs"] = {"start_method": "fork"}
    return make_vec_env(factory, n_envs=N_ENVS, seed=SEED, **kwargs)

class EmptyMPSCache(BaseCallback):
    def _on_step(self): return True
    def _on_rollout_end(self):
        if DEVICE == "mps": torch.mps.empty_cache()

# ───────── Optuna Objective ─────────
def objective(trial: optuna.Trial):
    hp = dict(
        learning_rate = trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        ent_coef      = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        gamma         = trial.suggest_float("gamma", 0.93, 0.999),
        gae_lambda    = trial.suggest_float("gae_lambda", 0.90, 0.98),
        vf_coef       = trial.suggest_float("vf_coef", 0.3, 1.0),
        clip_range    = trial.suggest_float("clip_range", 0.1, 0.3),
        n_epochs      = trial.suggest_categorical("n_epochs", [2, 4]),
    )
    env   = make_env(env_fn_random)  # Subproc
    model = MaskablePPO("MultiInputPolicy", env,
                        n_steps=2048, batch_size=512, seed=SEED,
                        device=DEVICE, verbose=0, **hp)

    eval_env = Monitor(RandomOpponent())
    stop_cb  = StopTrainingOnNoModelImprovement(3, 5, verbose=0)
    eval_cb  = EvalCallback(eval_env, n_eval_episodes=TRIAL_EP,
                            eval_freq=10_000, callback_after_eval=stop_cb,
                            verbose=0)

    model.learn(TRIAL_STEP, callback=CallbackList([eval_cb, EmptyMPSCache()]),
                progress_bar=False)
    return win_rate_vs_random(model)

# ───────── Phase-0 : Optuna → fine-tune ─────────
def phase0_optuna(trials: int, timeout: Optional[int]) -> Path:
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME,
                                storage=OPTUNA_DB, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=trials, timeout=timeout,
                   n_jobs=1, show_progress_bar=True)

    best = study.best_params
    env  = make_env(env_fn_random)  # Subproc
    model = MaskablePPO("MultiInputPolicy", env,
                        n_steps=2048, batch_size=512, seed=SEED,
                        device=DEVICE, verbose=1,
                        learning_rate=best.get("lr", 2.5e-4),
                        ent_coef    =best.get("ent_coef", 0.001),
                        gamma       =best.get("gamma", 0.99),
                        gae_lambda  =best.get("gae_lambda", 0.95),
                        vf_coef     =best.get("vf_coef", 0.5),
                        clip_range  =best.get("clip_range", 0.2),
                        n_epochs    =best.get("n_epochs", 4))

    model.learn(FINAL_STEP, callback=EmptyMPSCache(), progress_bar=True)
    model.save(BEST_MODEL)
    print(f"[Optuna] fine-tuned model saved → {BEST_MODEL}")
    return BEST_MODEL

# ───────── Self-play League ─────────
def league_train(base: Path, phases: int = 3):
    model = MaskablePPO.load(base, device=DEVICE)
    pool: List[Path] = [base]

    for phase in range(1, phases + 1):
        if phase == 1:  # vs Random
            env = make_env(env_fn_random, multiproc=True)
        else:           # vs pool (DummyVecEnv, プロセス無し)
            def single_env(op_path: Path):
                return ActionMasker(FixedOpponent(op_path), lambda e:e.action_masks())
            env_fns = [functools.partial(single_env, pool[np.random.randint(len(pool))])
                        for _ in range(N_ENVS)]
            env = DummyVecEnv(env_fns)

        model.set_env(env)
        model.learn(PHASE_STEP, reset_num_timesteps=False,
                    callback=EmptyMPSCache(), progress_bar=True)

        saved = SAVE_DIR / f"league_p{phase}.zip"
        model.save(saved); pool.append(saved)
        print(f"[League] phase-{phase} saved → {saved}")

# ───────── CLI ─────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--trials",  type=int, default=40, help="Optuna trial 数")
    pa.add_argument("--timeout", type=int, default=None, help="Optuna 制限秒 (none=無制限)")
    pa.add_argument("--skip_optuna", action="store_true", help="既存ベストモデルを再利用")
    pa.add_argument("--phases",  type=int, default=3, help="リーグフェーズ数")
    args = pa.parse_args()

    print("[Device]", DEVICE, "| envs =", N_ENVS)

    model_path = BEST_MODEL if (args.skip_optuna and BEST_MODEL.exists()) \
                else phase0_optuna(args.trials, args.timeout)
    league_train(model_path, args.phases)
