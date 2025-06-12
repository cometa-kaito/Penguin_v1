#!/usr/bin/env python3
# --------------------------------------------------
#  visualize_all_results.py  –  client と同観測で評価
# --------------------------------------------------
import glob, re, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

# ---------- 設定 ----------
EVAL_EPISODES = 300
RAND_SEED = 42
random.seed(RAND_SEED); np.random.seed(RAND_SEED)

# ---------- 観測フィルタ ----------
class ObsFilterWrapper(PenguinPartyGymEnv):
    """hand / board / current_player だけ返す"""
    def _get_observation(self):
        base = super()._get_observation()
        if base is None: return None
        return {"hand": base["hand"],
                "board": base["board"],
                "current_player": base["current_player"]}

    @property
    def current_player(self):
        return getattr(self, "env", self).current_player

    def outcome(self, idx:int=0):
        return getattr(self, "env", self).outcome(idx)

# ---------- Env factory ----------
def make_env(expect_mask: bool):
    base = ObsFilterWrapper()
    env  = ActionMasker(base, lambda e: e.action_masks()) if expect_mask else base
    return Monitor(env), base

def rand_act(mask: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(mask)))

def duel(model: MaskablePPO, expect_mask: bool, model_first: bool):
    env, base = make_env(expect_mask)
    obs,_ = env.reset(); done=False; tot_r=0.0
    while not done:
        mask = base.action_masks()
        turn = base.current_player
        action = (model.predict(obs, deterministic=True, action_masks=mask)[0]
                  if (turn == 0) == model_first else rand_act(mask))
        obs, r, term, trunc, _ = env.step(action)
        tot_r += r; done = term or trunc
    return tot_r, base.outcome(0 if model_first else 1)

def evaluate(path: str, n_episodes=EVAL_EPISODES):
    model = MaskablePPO.load(path)
    expect_mask = isinstance(model.observation_space, gym.spaces.Dict) \
                  and "action_mask" in model.observation_space.spaces
    win=draw=lose=0; rewards=[]
    for ep in range(n_episodes):
        tot_r,res = duel(model, expect_mask, model_first=(ep%2==0))
        rewards.append(tot_r)
        if   res=="win":  win+=1
        elif res=="draw": draw+=1
        else:             lose+=1
    return dict(mean_reward=round(np.mean(rewards),3),
                std_reward =round(np.std(rewards),3),
                win_rate   =round(win  / n_episodes*100,1),
                draw_rate  =round(draw / n_episodes*100,1),
                lose_rate  =round(lose / n_episodes*100,1))

# ---------- モデル一覧 ----------
globs = [
    "grid_models/*.zip",
    "local_rand/*.zip",
    "finetuning/*.zip",
    "finetuning/local_rand/*.zip",
    "models/*.zip",
    "models/best/*.zip",      # ★ 追加: best フォルダ
]
manual = [
    "finetuning/ppo_penguin_refined.zip",
    "finetuning/ppo_penguin_tuned.zip",
    "finetuning/ppo_penguin_league.zip",
]
model_files = sorted({p for g in globs for p in glob.glob(g)} |
                     {p for p in manual if Path(p).is_file()})
if not model_files:
    raise FileNotFoundError("No model zip found")
print(f"Found {len(model_files)} model(s)")

# ---------- 評価ループ ----------
records=[]; t0=time.time()
for i,p in enumerate(model_files,1):
    print(f"[{i}/{len(model_files)}] {p} ... ", end="", flush=True)
    met = evaluate(p); print("done")
    name = Path(p).stem
    m = re.search(r"lr([0-9e\.\-]+)_ent([0-9\.]+)", name)
    lr  = float(m.group(1)) if m else None
    ent = float(m.group(2)) if m else None
    records.append({"model": name, "lr": lr, "ent_coef": ent, **met, "path": p})

print(f"\nFinished in {(time.time()-t0)/60:.1f} min")

df = (pd.DataFrame(records)
        .sort_values(["mean_reward","win_rate"], ascending=[False,False])
        .reset_index(drop=True))
print("\n===== Summary ====="); print(df)
df.to_csv("summary.csv", index=False)
print("Saved to summary.csv")
