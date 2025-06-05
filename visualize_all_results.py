# --------------------------------------------------
#  visualize_all_results.py     (finetune_and_search.py と同じ階層)
# --------------------------------------------------
# 目的:
#   ・事前学習モデル / 追加学習モデル / ランダムサーチ結果を一括評価し、
#     平均報酬・勝率などの主要指標を比較する。
#   ・結果を DataFrame で一覧表示し、summary.csv として保存する。
# --------------------------------------------------

import glob, re, time
from pathlib import Path

import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from penguin_party_gym import PenguinPartyGymEnv

# ------------------ 評価ハイパーパラメータ ------------------
EVAL_EPISODES = 200   # 1 モデルあたりの評価エピソード数
# -----------------------------------------------------------

def evaluate_model(model_path: str, n_episodes: int = EVAL_EPISODES) -> dict:
    """
    指定モデルを n_episodes 回プレイし、主要指標を dict で返す
      - mean_reward : 総報酬の平均
      - std_reward  : 総報酬の標準偏差
      - win_rate    : 勝率 (%)
      - draw_rate   : 引き分け率 (%)
      - remain_avg  : 残り手札枚数の平均
    """
    base_env = PenguinPartyGymEnv()     # 生環境
    env = Monitor(base_env)             # 総報酬を自動合算

    model = MaskablePPO.load(model_path, env=env)

    total_rewards, remain_cards = [], []
    wins, draws = 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0

        # 1 エピソード
        while not done:
            mask = base_env.action_masks()                  # マスク取得
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward

        # 勝敗判定 (自分: index 0, 相手: index 1)
        scores = base_env.env.get_scores()
        if scores[0] < scores[1]:
            wins += 1
        elif scores[0] == scores[1]:
            draws += 1

        total_rewards.append(ep_reward)
        remain_cards.append(scores[0])

    return {
        "mean_reward": round(float(np.mean(total_rewards)), 3),
        "std_reward":  round(float(np.std(total_rewards)), 3),
        "win_rate":    round(wins / n_episodes * 100, 1),
        "draw_rate":   round(draws / n_episodes * 100, 1),
        "remain_avg":  round(float(np.mean(remain_cards)), 2),
    }

# ------------------ 評価対象モデルの収集 ------------------
model_files: list[str] = []

# (1) 既存グリッド／ローカルサーチ
model_files += glob.glob("grid_models/*.zip")
model_files += glob.glob("local_rand/*.zip")

# (2) finetuning フォルダ
model_files += glob.glob("finetuning/*.zip")
model_files += glob.glob("finetuning/local_rand/*.zip")

# (3) 互換用 : 手動コピー分 (重複 OK, 後で集合で排除)
model_files += [
    "finetuning/ppo_penguin_refined.zip",
    "finetuning/ppo_penguin_tuned.zip",
    "finetuning/ppo_penguin_league.zip",
]

# ---------- 重複排除 & 実在チェック ----------
model_files = sorted({p for p in model_files if Path(p).is_file()})
if not model_files:
    raise FileNotFoundError("評価対象となる .zip モデルが見つからない。パス設定を確認せよ。")

print(f"Found {len(model_files)} model(s) to evaluate ...")

# ------------------ ループ評価 ------------------
records = []                         # 評価結果を蓄積
start_time = time.time()

for idx, path in enumerate(model_files, 1):
    print(f"[{idx}/{len(model_files)}] evaluating {path} ... ", end="")
    metrics = evaluate_model(path)
    print("done")

    # ファイル名パターンからハイパーパラメータを抽出 (存在すれば)
    name = Path(path).stem
    m = re.search(r"lr([0-9e\.\-]+)_ent([0-9\.]+)", name)
    lr  = float(m.group(1)) if m else None
    ent = float(m.group(2)) if m else None

    records.append({
        "model": name,
        "lr": lr,
        "ent_coef": ent,
        **metrics,
        "path": path,
    })

elapsed = time.time() - start_time
print(f"\nEvaluation finished in {elapsed/60:.1f} min")

# ------------------ DataFrame & 出力 ------------------
df = (
    pd.DataFrame(records)
      .sort_values(by=["mean_reward", "win_rate"], ascending=[False, False])
      .reset_index(drop=True)
)

print("\n===== Summary (sorted by mean_reward then win_rate) =====")
print(df)

# CSV 保存
df.to_csv("summary.csv", index=False)
print("\nCSV saved -> summary.csv")
