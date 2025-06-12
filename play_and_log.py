#!/usr/bin/env python3
"""
play_and_log.py

指定した Maskable-PPO モデルを 1 対戦ずつ実行し、すべての手を
テキストでログ出力して勝敗集計を確認するデバッグ用スクリプト。
"""

import argparse, random, numpy as np
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from penguin_party_gym import PenguinPartyGymEnv

# ──────────────────────────────────────────────────────────────
def make_env(expect_mask: bool):
    base = PenguinPartyGymEnv()                             # seed 不要
    env  = ActionMasker(base, lambda e: e.action_masks()) if expect_mask else base
    return env, base

def rand_act(mask: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(mask)))

# ──────────────────────────────────────────────────────────────
def play_one(model: MaskablePPO, episode_idx: int, model_first: bool):
    """
    1 エピソードをプレイしてログを出力。
    model_first=True なら player0 がモデル、False なら player1 がモデル。
    """
    expect_mask = isinstance(model.observation_space, gym.spaces.Dict) \
                  and "action_mask" in model.observation_space.spaces
    env, base   = make_env(expect_mask)

    obs, _ = env.reset()
    done   = False
    turn   = 0

    print(f"\n=== Episode {episode_idx+1}  "
          f'({"model先手" if model_first else "model後手"}) ===')

    while not done:
        mask = base.action_masks()
        ply  = base.env.current_player

        if (ply == 0) == model_first:               # モデルの手番？
            action_idx, _ = model.predict(obs, deterministic=True,
                                          action_masks=mask)
            who = "MODEL "
        else:
            action_idx    = rand_act(mask)
            who = "RANDOM"

        act = base.action_map[int(action_idx)]
        print(f"Turn {turn:2d} | P{ply}={who}: {act}")
        obs, r, term, trunc, _ = env.step(action_idx)
        done = term or trunc
        turn += 1

    # --- ゲーム終了 ---
    scores = base.env.get_scores()
    outcome = base.env.outcome(0 if model_first else 1)
    print(f"END | scores={scores}  →  outcome={outcome}\n")
    return outcome

# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",
        help="評価したい .zip モデルファイル")
    parser.add_argument("-n", "--episodes", type=int, default=4,
        help="実行エピソード数（偶数推奨）")
    args = parser.parse_args()

    if not Path(args.model_path).is_file():
        raise FileNotFoundError(args.model_path)

    model = MaskablePPO.load(args.model_path)
    random.seed(2025); np.random.seed(2025)

    win=draw=lose=0
    for ep in range(args.episodes):
        res = play_one(model, ep, model_first=(ep%2==0))
        if   res=="win":  win+=1
        elif res=="draw": draw+=1
        else:             lose+=1

    print("#"*40)
    print(f"集計  (total={args.episodes})")
    print(f" wins : {win}  ({win/args.episodes*100:4.1f}%)")
    print(f" draws: {draw}  ({draw/args.episodes*100:4.1f}%)")
    print(f" loses: {lose}  ({lose/args.episodes*100:4.1f}%)")
    print("#"*40)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
