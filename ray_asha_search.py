#!/usr/bin/env python3
import argparse, warnings, numpy as np, os
warnings.filterwarnings("ignore", category=UserWarning)

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session                        # ← ここだけで OK

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from penguin_party_gym import PenguinPartyGymEnv


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


def train_fn(cfg):
    env   = make_vec_env(PenguinPartyGymEnv, n_envs=8)
    model = MaskablePPO("MultiInputPolicy", env,
                        learning_rate=cfg["lr"],
                        ent_coef=cfg["ent"],
                        n_steps=1024, batch_size=256, verbose=0)

    for _ in range(0, 600_000, 100_000):
        model.learn(total_timesteps=100_000, reset_num_timesteps=False)
        session.report(
            {"win_rate": evaluate(model),
             "timesteps": model.num_timesteps}
        )


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--samples", type=int, default=60)
    args = pa.parse_args()

    ray.init(ignore_reinit_error=True)

    search = {
        "lr" : tune.loguniform(1e-5, 5e-4),
        "ent": tune.loguniform(1e-4, 1e-2),
    }

    scheduler = ASHAScheduler(
        max_t=600_000,
        grace_period=200_000,
        reduction_factor=3,
        time_attr="timesteps",
        metric="win_rate",
        mode="max",
    )

    tune.run(
        train_fn,
        config=search,
        num_samples=args.samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 0},
    )


if __name__ == "__main__":
    main()
