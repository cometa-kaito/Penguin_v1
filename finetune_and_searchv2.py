from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
from collections import deque

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from penguin_party_gym import PenguinPartyGymEnv

# --- ディレクトリ設定 ---
FINETUNE_DIR = Path("finetuning")
LOCAL_RAND_DIR = FINETUNE_DIR / "local_rand"
LOG_DIR = Path("logs")
FINETUNE_DIR.mkdir(exist_ok=True)
LOCAL_RAND_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

BASE_MODEL_PATH = Path("models/optuna_refit.zip")
FINE_TUNE_STEPS = 200_000
REFINED_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_refined.zip"

TUNED_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_tuned.zip"
REWARD_TUNE_STEPS = 200_000
TUNED_REWARD_CONFIG = {
    "valid_action_bonus": 0,
    "win_base": 10.0,
    "win_per_diff": 20.0,
    "lose_base": -10.0,
    "lose_per_diff": 20.0,
    "invalid_action_penalty": -0.8,
}

LEAGUE_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_league.zip"
SELF_PLAY_STEPS = 200_000

SEARCH_SPACE = {"lr": [3e-5, 7e-5], "ent": [0.002, 0.003, 0.004]}
SAMPLE_STEPS = 200_000
EVAL_EPISODES = 100
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MOVING_AVG_WINDOW = 100

# --------- 評価関数 -----------
def make_env(expect_mask: bool):
    base = PenguinPartyGymEnv()
    env  = ActionMasker(base, lambda e: e.action_masks()) if expect_mask else base
    return Monitor(env), base

def rand_act(mask: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(mask)))

def duel(model: MaskablePPO, expect_mask: bool, model_first: bool) -> tuple:
    """1ゲーム分プレイし (総報酬, 勝敗, 残カード枚数, 手数) を返す"""
    env, base = make_env(expect_mask)
    obs, _ = env.reset()
    done, tot_r, steps = False, 0.0, 0

    while not done:
        mask = base.action_masks()
        turn = base.env.current_player
        if (turn == 0) == model_first:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        else:
            action = rand_act(mask)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
        tot_r += r
        steps += 1

    result = base.env.outcome(0 if model_first else 1)
    remain_cards_obj = base.env.hands[0 if model_first else 1]
    # ここがポイント！手札の枚数を数える
    if isinstance(remain_cards_obj, (list, tuple, np.ndarray)):
        remain_cards = len(remain_cards_obj)
    else:
        remain_cards = int(remain_cards_obj)
    return tot_r, result, remain_cards, steps




def evaluate(model, n_episodes=EVAL_EPISODES):
    import gymnasium as gym
    expect_mask = isinstance(model.observation_space, gym.spaces.Dict) \
                  and "action_mask" in model.observation_space.spaces
    rewards, remain_cards, ep_lens = [], [], []
    win = draw = lose = 0
    for ep in range(n_episodes):
        tot_r, res, remain, steps = duel(model, expect_mask, model_first=(ep % 2 == 0))
        rewards.append(tot_r)
        remain_cards.append(remain)
        ep_lens.append(steps)
        if   res == "win":  win  += 1
        elif res == "draw": draw += 1
        else:               lose += 1
    n = n_episodes
    return {
        "mean_reward"      : round(float(np.mean(rewards)),3),
        "std_reward"       : round(float(np.std(rewards)),3),
        "moving_avg_reward": round(float(pd.Series(rewards).rolling(MOVING_AVG_WINDOW, min_periods=1).mean().iloc[-1]), 3),
        "win_rate"         : round(win  / n * 100, 1),
        "draw_rate"        : round(draw / n * 100, 1),
        "lose_rate"        : round(lose / n * 100, 1),
        "avg_remain"       : round(np.mean(remain_cards), 3),
        "std_remain"       : round(np.std(remain_cards), 3),
        "avg_ep_len"       : round(np.mean(ep_lens), 2),
        "std_ep_len"       : round(np.std(ep_lens), 2),
    }

# --------- ログ用コールバック -----------
class ProgressLogger(BaseCallback):
    def __init__(self, log_path, save_prefix, eval_fn, eval_interval=50000, verbose=1):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.save_prefix = save_prefix
        self.eval_fn = eval_fn
        self.eval_interval = eval_interval
        self.records = []
        self.reward_history = deque(maxlen=MOVING_AVG_WINDOW)

    def _on_rollout_end(self) -> None:
        # rollout毎（n_stepsごと）に最新の報酬を保存
        if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
            ep_rew_mean = self.model.logger.name_to_value["rollout/ep_rew_mean"]
            self.reward_history.append(ep_rew_mean)

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if steps % self.eval_interval == 0 or steps == self.model._total_timesteps:
            model_path = self.log_path / f"{self.save_prefix}_{steps//1000}k.zip"
            self.model.save(model_path)
            metrics = self.eval_fn(self.model)
            # PPO内部指標
            log_dict = self.model.logger.name_to_value
            metrics['clip_fraction']   = log_dict.get("train/clip_fraction", np.nan)
            metrics['value_loss']      = log_dict.get("train/value_loss", np.nan)
            metrics['policy_loss']     = log_dict.get("train/policy_loss", np.nan)
            metrics['entropy_loss']    = log_dict.get("train/entropy_loss", np.nan)
            metrics['approx_kl']       = log_dict.get("train/approx_kl", np.nan)
            metrics['learning_rate']   = log_dict.get("train/learning_rate", np.nan)
            # 報酬の進捗移動平均（学習log由来・評価由来で両方入れてOK）
            metrics['rollout_ep_rew_mean'] = np.mean(self.reward_history) if self.reward_history else np.nan
            metrics['steps'] = steps
            self.records.append(metrics)
            if self.verbose:
                if self.verbose:
                    print(f"[{self.save_prefix}] Step {steps}:")
                    for k, v in metrics.items():
                        print(f"    {k}: {v}")
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.records)
        csv_path = self.log_path / f"{self.save_prefix}_curve.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if self.verbose:
            print(f"Progress log saved: {csv_path}")

# --- 安全な環境作成 ---
def make_safe_env(**kwargs):
    return PenguinPartyGymEnv(**kwargs)

# =========== 微調整 ============
def fine_tune_base_model():
    print(f"\n① Fine-tune: {BASE_MODEL_PATH.name} ({FINE_TUNE_STEPS:,} steps)")
    env = make_vec_env(lambda: make_safe_env(), n_envs=16)
    model = MaskablePPO.load(
        BASE_MODEL_PATH,
        env=env,
        learning_rate=5e-5,
        ent_coef=0.001,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    logger = ProgressLogger(LOG_DIR, "finetune", evaluate, eval_interval=50_000)
    model.learn(total_timesteps=FINE_TUNE_STEPS, callback=logger, progress_bar=True)
    model.save(REFINED_MODEL_PATH)
    print(f"➔ saved → {REFINED_MODEL_PATH}")

# =========== 報酬チューニング ============
def reward_tuning():
    print("\n② Reward tuning")
    env = make_vec_env(lambda: make_safe_env(reward_config=TUNED_REWARD_CONFIG), n_envs=16)
    model = MaskablePPO.load(
        REFINED_MODEL_PATH,
        env=env,
        learning_rate=5e-5,
        ent_coef=0.001,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    logger = ProgressLogger(LOG_DIR, "reward_tune", evaluate, eval_interval=50_000)
    model.learn(total_timesteps=REWARD_TUNE_STEPS, callback=logger, progress_bar=True)
    model.save(TUNED_MODEL_PATH)
    print(f"➔ saved → {TUNED_MODEL_PATH}")

# =========== 自己対戦リーグ ============
def self_play_league():
    print("\n③ Self-play league")
    opponent = MaskablePPO.load(BASE_MODEL_PATH)
    class SelfPlayEnv(PenguinPartyGymEnv):
        def step(self, action):
            obs, rew, term, trunc, info = super().step(action)
            if not (term or trunc):
                mask = self.action_masks()
                opp_action, _ = opponent.predict(obs, deterministic=True, action_masks=mask)
                obs, opp_rew, term, trunc, info = super().step(opp_action)
                rew -= opp_rew
            return obs, rew, term, trunc, info

    env = make_vec_env(SelfPlayEnv, n_envs=8)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=5e-5,
        ent_coef=0.001,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )
    logger = ProgressLogger(LOG_DIR, "selfplay_league", evaluate, eval_interval=50_000)
    model.learn(total_timesteps=SELF_PLAY_STEPS, callback=logger, progress_bar=True)
    model.save(LEAGUE_MODEL_PATH)
    print(f"➔ saved → {LEAGUE_MODEL_PATH}")

# =========== 局所ランダム探索 ============
def local_random_search():
    print("\n④ Local random search")
    results = []
    for lr, ent in product(SEARCH_SPACE["lr"], SEARCH_SPACE["ent"]):
        print(f"Training lr={lr}, ent_coef={ent}")
        env = make_vec_env(PenguinPartyGymEnv, n_envs=8)
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            ent_coef=ent,
            n_steps=2048,
            batch_size=512,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=0,
        )
        prefix = f"localsearch_lr{lr}_ent{ent}"
        logger = ProgressLogger(LOG_DIR, prefix, evaluate, eval_interval=50_000)
        model.learn(total_timesteps=SAMPLE_STEPS, callback=logger)
        metrics = evaluate(model)
        path = LOCAL_RAND_DIR / f"ppo_lr{lr}_ent{ent}.zip"
        model.save(path)
        results.append({
            "lr": lr,
            "ent_coef": ent,
            **metrics,
            "path": str(path)
        })
        print(f"lr={lr}, ent_coef={ent} → {metrics}")

    df = pd.DataFrame(results).sort_values(by="mean_reward", ascending=False)
    print(df)
    df.to_csv(LOCAL_RAND_DIR / "local_search_results.csv", index=False, encoding="utf-8-sig")

# --- 実行フロー ---
def main():
    fine_tune_base_model()
    reward_tuning()
    self_play_league()
    #local_random_search()

if __name__ == "__main__":
    main()
