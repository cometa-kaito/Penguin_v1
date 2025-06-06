"""
変更点:
 1. PenguinPartyGymEnv(**env_kwargs) を使えるようにして報酬設定を正しく反映
 2. make_tuned_env() で reward_config を渡す
"""

import random, itertools
from pathlib import Path
import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from penguin_party_gym import PenguinPartyGymEnv
from penguin_party_env import PenguinPartyEnv  # ← 報酬定数参照のみ

# ---------- ディレクトリ準備 ----------
FINETUNE_DIR   = Path("finetuning")
LOCAL_RAND_DIR = FINETUNE_DIR / "local_rand"
FINETUNE_DIR.mkdir(exist_ok=True)
LOCAL_RAND_DIR.mkdir(exist_ok=True)

# ================================================== #
# ① Fine-tune
# ================================================== #
BASE_MODEL          = "grid_models/ppo_lr5e-05_ent0.001.zip"
MORE_STEPS          = 1_000_000
REFINED_MODEL_PATH  = FINETUNE_DIR / "ppo_penguin_refined.zip"

env_ft = PenguinPartyGymEnv()
model  = MaskablePPO.load(BASE_MODEL, env=env_ft, learning_rate=5e-5)

print(f"[Step 1] Fine-tuning {BASE_MODEL} for {MORE_STEPS:,} steps …")
model.learn(total_timesteps=MORE_STEPS, progress_bar=True)
model.save(REFINED_MODEL_PATH)
print(f"⸺ saved to {REFINED_MODEL_PATH}")

# ================================================== #
# ② Reward tuning
# ================================================== #
TUNED_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_tuned.zip"

tuned_reward = {
    "valid_action_bonus":   0.02,
    "win_base":             2.0,
    "win_per_diff":         0.2,
    "lose_base":           -2.0,
    "lose_per_diff":        0.2,
    "invalid_action_penalty": -0.8,
}

def make_tuned_env():
    return PenguinPartyGymEnv(reward_config=tuned_reward)

env_tuned = make_vec_env(make_tuned_env, n_envs=8)
model_tuned = MaskablePPO.load(REFINED_MODEL_PATH, env=env_tuned, learning_rate=5e-5)

print("[Step 2] Continue training with tuned reward schedule …")
model_tuned.learn(total_timesteps=10_000, progress_bar=True)
model_tuned.save(TUNED_MODEL_PATH)
print(f"⸺ saved to {TUNED_MODEL_PATH}")

# ================================================== #
# ③ Self-play league
# ================================================== #
LEAGUE_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_league.zip"
model_safe = MaskablePPO.load("grid_models/ppo_lr5e-05_ent0.001.zip", env=None)
model_aggr = MaskablePPO.load("grid_models/ppo_lr5e-05_ent0.01.zip",  env=None)

class SelfPlayEnv(PenguinPartyGymEnv):
    def __init__(self, opponent):
        super().__init__()
        self.opponent_model = opponent

    def step(self, action_idx):
        obs, reward, done, trunc, info = super().step(action_idx)
        if done:
            return obs, reward, done, trunc, info

        mask = self.action_masks()
        opp_action, _ = self.opponent_model.predict(obs, deterministic=True, action_masks=mask)
        obs, opp_reward, done, trunc, info = super().step(opp_action)
        reward -= opp_reward
        return obs, reward, done, trunc, info

def make_selfplay_env():
    opp = random.choice([model_safe, model_aggr])
    return SelfPlayEnv(opp)

vec_sp = make_vec_env(make_selfplay_env, n_envs=8)
league_model = MaskablePPO("MultiInputPolicy", vec_sp, learning_rate=5e-5, verbose=1)

print("[Step 3] Training in self-play league …")
league_model.learn(total_timesteps=10_000, progress_bar=True)
league_model.save(LEAGUE_MODEL_PATH)
print(f"⸺ saved to {LEAGUE_MODEL_PATH}")

# ================================================== #
# ④ Local random search
# ================================================== #
search_space = {"lr": [3e-5, 7e-5], "ent": [0.002, 0.003, 0.004]}
SAMPLE_STEPS, EVAL_N = 10_000, 100
records = []

def quick_eval(m):
    env = Monitor(PenguinPartyGymEnv())
    wins, total_r = 0, []
    for _ in range(EVAL_N):
        o, _ = env.reset()
        done, r = False, 0.0
        while not done:
            a, _ = m.predict(o, deterministic=True, action_masks=env.unwrapped.action_masks())
            o, rew, term, trunc, _ = env.step(a)
            done = term or trunc
            r += rew
        scores = env.env.env.get_scores()
        if scores[0] < scores[1]:
            wins += 1
        total_r.append(r)
    return np.mean(total_r), wins / EVAL_N * 100

for lr, ent in itertools.product(search_space["lr"], search_space["ent"]):
    env = make_vec_env(PenguinPartyGymEnv, n_envs=4)
    m = MaskablePPO("MultiInputPolicy", env, learning_rate=lr, ent_coef=ent,
                    n_steps=1024, batch_size=256, verbose=0)
    m.learn(total_timesteps=SAMPLE_STEPS)
    mean_r, win = quick_eval(m)

    path = LOCAL_RAND_DIR / f"ppo_lr{lr}_ent{ent}.zip"
    m.save(path)
    records.append((lr, ent, round(mean_r, 3), round(win, 1), str(path)))
    print(f"lr={lr:.0e}, ent={ent:.3f} → reward {mean_r:.2f}, win {win:.1f}%")

df = (
    pd.DataFrame(records, columns=["lr", "ent", "meanR", "win%", "path"])
      .sort_values("meanR", ascending=False)
)
print("\n========= Local Random Search Result =========")
print(df)
