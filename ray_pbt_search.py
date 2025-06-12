#!/usr/bin/env python3
"""
Population-Based Training (Ray 2.10 compatible)
  * 8 並列個体
  * 1.2 M step まで進化
  * Checkpoint は tempfile に書き出して渡す
"""

import warnings, numpy as np, os, tempfile, shutil
warnings.filterwarnings("ignore", category=UserWarning)

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.air import session
try:
    from ray.train import Checkpoint         # Ray ≤2.11/2.10
except ImportError:
    from ray.air import Checkpoint           # Ray ≥2.12

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from penguin_party_gym import PenguinPartyGymEnv


# ─────────── 評価関数 ───────────
def evaluate(model, n_ep=20):
    env, wins = PenguinPartyGymEnv(), 0
    for _ in range(n_ep):
        obs, _ = env.reset(); done = False
        while not done:
            mask = env.unwrapped.action_masks()
            act, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        wins += (env.env.outcome() == "win")
    return wins / n_ep * 100


# ─────────── 1 個体の学習ループ ───────────
def train_fn(cfg):
    env   = make_vec_env(PenguinPartyGymEnv, n_envs=8)
    model = MaskablePPO("MultiInputPolicy", env,
                        learning_rate=cfg["lr"],
                        ent_coef=cfg["ent"],
                        n_steps=1024, batch_size=256, verbose=0)

    while True:
        model.learn(total_timesteps=50_000, reset_num_timesteps=False)

        win = evaluate(model)
        # --- 一時ディレクトリへ保存 → Checkpoint 化 -------
        tmpdir = tempfile.mkdtemp()
        model.save(os.path.join(tmpdir, "model.zip"))
        ckpt = Checkpoint.from_directory(tmpdir)

        session.report(
            {"win_rate": win,
             "timesteps": model.num_timesteps,
             "lr": cfg["lr"], "ent": cfg["ent"]},
            checkpoint=ckpt,
        )
        shutil.rmtree(tmpdir)   # 後片付け


# ─────────── メイン ───────────
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    search_space = {
        "lr" : tune.loguniform(1e-5, 5e-4),
        "ent": tune.loguniform(1e-4, 1e-2),
    }

    pbt = PopulationBasedTraining(
        time_attr="timesteps",
        metric="win_rate",
        mode="max",
        perturbation_interval=100_000,
        hyperparam_mutations={
            "lr" : lambda: float(10 ** np.random.uniform(-5, -3.3)),
            "ent": lambda: float(10 ** np.random.uniform(-4, -1.7)),
        },
    )

    tune.run(
        train_fn,
        name="pbt_penguin_party",
        scheduler=pbt,
        num_samples=8,                     # 個体数
        resources_per_trial={"cpu": 4, "gpu": 0},
        stop={"timesteps": 1_200_000},    # 1.2 M step で打ち切り
        config=search_space,
    )
