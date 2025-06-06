# 変更点:
#   ・env.unwrapped.action_masks() で安全にマスクを取得
#   ・その他ロジックはそのまま
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

learning_rates = [2.5e-4, 1e-4, 5e-5]
ent_coefs      = [0.01, 0.005, 0.001]

TOTAL_STEPS    = 2_000_000
EVAL_EPISODES  = 200
N_ENVS         = 16

results = []
save_dir = Path("grid_models")
save_dir.mkdir(exist_ok=True)

def evaluate(model, n_episodes=200):
    env = Monitor(PenguinPartyGymEnv())
    tot_rewards, remain_cards = [], []
    wins, draws = 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            mask = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward

        scores = env.env.env.get_scores()      # (self, opponent)
        if scores[0] < scores[1]:
            wins += 1
        elif scores[0] == scores[1]:
            draws += 1

        tot_rewards.append(ep_reward)
        remain_cards.append(scores[0])

    mean_r = np.mean(tot_rewards)
    std_r  = np.std(tot_rewards)
    win_r  = wins / n_episodes
    return mean_r, std_r, win_r, draws / n_episodes, np.mean(remain_cards)

# --- グリッドサーチ本体 ---
for lr, ent in product(learning_rates, ent_coefs):
    print(f"\n=== Training lr={lr}, ent_coef={ent} ===")
    vec_env = make_vec_env(PenguinPartyGymEnv, n_envs=N_ENVS)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=lr,
        ent_coef=ent,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)

    model_path = save_dir / f"ppo_lr{lr}_ent{ent}.zip"
    model.save(model_path)

    mean_r, std_r, win_r, draw_r, mean_rem = evaluate(model, n_episodes=EVAL_EPISODES)
    results.append({
        "lr": lr, "ent_coef": ent,
        "mean_reward": round(mean_r, 3),
        "std_reward":  round(std_r, 3),
        "win_rate":    round(win_r * 100, 1),
        "draw_rate":   round(draw_r * 100, 1),
        "remain_avg":  round(mean_rem, 2),
        "model_path":  str(model_path),
    })

df = pd.DataFrame(results).sort_values(by="mean_reward", ascending=False)
print("\n===== Grid Search Result (sorted by mean_reward) =====")
print(df)

csv_path = save_dir / "grid_search_results.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✓ CSV saved → {csv_path}")
