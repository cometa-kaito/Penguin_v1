#!/usr/bin/env python3
"""
短時間（1e4 step）の学習を行い、エピソード毎に
・最終報酬
・rollout/ep_rew_mean
を出力して報酬が届いているか確認する。
"""
import numpy as np
from penguin_party_gym import PenguinPartyGymEnv     # 既存 Gym ラッパーで OK
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# ---------- 学習パラメータ ----------
TOTAL_STEPS = 10_000     # デバッグ用に小さく
N_ENVS      = 4
EVAL_EPIS   = 5

# ---------- 環境 ----------
def make_env():
    return Monitor(PenguinPartyGymEnv())   # reward_config はデフォルトで per_card_bonus 込み

vec_env = make_vec_env(make_env, n_envs=N_ENVS)

# ---------- モデル ----------
model = MaskablePPO(
    "MultiInputPolicy",
    vec_env,
    learning_rate=2.5e-4,
    ent_coef=0.01,
    n_steps=512,
    batch_size=512,
    verbose=0,
)
print("=== 学習開始 ===")
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
print("=== 学習終了 ===\n")

# ---------- 評価 ----------
env = Monitor(PenguinPartyGymEnv())
for ep in range(1, EVAL_EPIS + 1):
    obs, _ = env.reset()
    done, ep_r = False, 0.0
    while not done:
        mask = env.unwrapped.action_masks()
        act, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        ep_r += r
    print(f"[Episode {ep}]  Total reward = {ep_r:.3f}  |  outcome = {env.env.env.outcome()}")
