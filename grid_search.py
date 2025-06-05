# ==================================================
#  Penguin Party  ─ PPO グリッドサーチスクリプト
# --------------------------------------------------
# ・学習率 × エントロピー係数 の組み合わせを総当たり
# ・各モデルを TOTAL_STEPS 学習 → EVAL_EPISODES で性能評価
# ・結果を DataFrame で表示し、CSV に保存
# ==================================================

# --------------------------------------------------
#  検証設定
# --------------------------------------------------
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

# ---------- グリッド探索対象ハイパーパラメータ ----------
learning_rates = [2.5e-4, 1e-4, 5e-5]   # 学習率 (lr)
ent_coefs      = [0.01, 0.005, 0.001]   # エントロピー係数 (探索度)

TOTAL_STEPS    = 2_000_000      # 1 組あたりの学習ステップ数
EVAL_EPISODES  = 200         # 評価エピソード数
N_ENVS         = 16           # 並列環境数 (CPU に余裕があれば増やす)

results = []                                       # すべての評価結果を格納
save_dir = Path("grid_models")                     # モデル保存先
save_dir.mkdir(exist_ok=True)

# --------------------------------------------------
#  評価関数 (マスク対応版)
# --------------------------------------------------
def evaluate(model, n_episodes=200):
    """
    与えられたモデルを n_episodes 回プレイし主要指標を返す。
      - mean_r, std_r : 総報酬の平均・標準偏差
      - win_rate      : 勝率 (0‒1)
      - draw_rate     : 引き分け率 (0‒1)
      - mean_rem      : 自分の残りカード枚数平均 (少ないほど良い)
    """
    env = Monitor(PenguinPartyGymEnv())
    tot_rewards, remain_cards = [], []
    wins, draws = 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            mask = env.env.action_masks()  # 有効手マスク
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward

        core = env.env.env                     # PenguinPartyEnv まで到達
        scores = core.get_scores()             # (self, opponent)
        if scores[0] < scores[1]:
            wins += 1
        elif scores[0] == scores[1]:
            draws += 1

        tot_rewards.append(ep_reward)
        remain_cards.append(scores[0])

    mean_r, std_r = np.mean(tot_rewards), np.std(tot_rewards)
    win_rate      = wins / n_episodes
    mean_rem      = np.mean(remain_cards)
    return mean_r, std_r, win_rate, draws / n_episodes, mean_rem


# --------------------------------------------------
#  グリッドサーチ本体
# --------------------------------------------------
for lr, ent in product(learning_rates, ent_coefs):
    print(f"\n=== Training lr={lr}, ent_coef={ent} ===")

    # --- 環境生成 (並列) ---
    vec_env = make_vec_env(PenguinPartyGymEnv, n_envs=N_ENVS)

    # --- モデル初期化 ---
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=lr,
        ent_coef=ent,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
        tensorboard_log=None,  # ← ログ不要なら None
    )

    # --- 学習 ---
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)

    # --- 保存 ---
    model_path = save_dir / f"ppo_lr{lr}_ent{ent}.zip"
    model.save(model_path)

    # --- 評価 ---
    mean_r, std_r, win_r, draw_r, mean_rem = evaluate(model, n_episodes=EVAL_EPISODES)

    # --- 結果をリストに追加 ---
    results.append({
        "lr": lr,
        "ent_coef": ent,
        "mean_reward": round(mean_r, 3),
        "std_reward": round(std_r, 3),
        "win_rate": round(win_r * 100, 1),
        "draw_rate": round(draw_r * 100, 1),
        "remain_avg": round(mean_rem, 2),
        "model_path": str(model_path),
    })

# --------------------------------------------------
#  結果表示 & CSV 保存
# --------------------------------------------------
df = pd.DataFrame(results).sort_values(by="mean_reward", ascending=False)

print("\n===== Grid Search Result (sorted by mean_reward) =====")
print(df)

# --- CSV 出力 ---
csv_path = save_dir / "grid_search_results.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✓ CSV を保存しました → {csv_path}")
