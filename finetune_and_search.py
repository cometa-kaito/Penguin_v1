# ==================================================
#  finetune_and_search.py   ← このファイル
# --------------------------------------------------
# Penguin Party 強化学習エージェントの
#   ① 追加学習
#   ② 報酬チューニング
#   ③ 自己対戦リーグ学習
#   ④ 局所ランダムハイパーパラメータ探索
# をワンストップで実行するスクリプト
# ==================================================

# ├─ finetuning/           ← ここに新しく生成されるモデルを保存
# │   ├─ ppo_penguin_refined.zip   … ①で出力
# │   ├─ ppo_penguin_tuned.zip     … ②で出力
# │   ├─ ppo_penguin_league.zip    … ③で出力
# │   └─ local_rand/               … ④のランダムサーチ結果
# │       └─ ppo_lr*_ent*.zip
# └─ grid_models/                  ← 既存ベースモデルの置き場
#     └─ ppo_lr5e-05_ent0.001.zip  など
# ==================================================

# --------------------------------------------------
#  共通 import & util
# --------------------------------------------------
import numpy as np, random, itertools
from pathlib import Path

# SB3 (Stable-Baselines3) 拡張版のマスク付き PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# 自作 Gym ラッパ & 生環境
from penguin_party_gym import PenguinPartyGymEnv
from penguin_party_env import PenguinPartyEnv

# ---------- 生成先フォルダを準備 ----------
FINETUNE_DIR      = Path("finetuning")
LOCAL_RAND_DIR    = FINETUNE_DIR / "local_rand"
FINETUNE_DIR.mkdir(exist_ok=True)
LOCAL_RAND_DIR.mkdir(exist_ok=True)

# ==================================================
# ① 追加学習：ベストモデルに上乗せ
# ==================================================
BEST_MODEL          = "grid_models/ppo_lr5e-05_ent0.001.zip"  # ベースとなる既存モデル
MORE_STEPS          = 1_000_000                                  # 追加学習ステップ
REFINED_MODEL_PATH  = FINETUNE_DIR / "ppo_penguin_refined.zip"

env = PenguinPartyGymEnv()  # 単一環境で OK（追加学習なので低負荷）

# 既存モデルを読み込み、学習率のみ上書き
model = MaskablePPO.load(
    BEST_MODEL,
    env=env,
    learning_rate=5e-5        # ★ Fine-tune 用に再指定
)

print(f"[Step 1] Fine-tuning {BEST_MODEL} for {MORE_STEPS:,} steps …")
model.learn(total_timesteps=MORE_STEPS, progress_bar=True)
model.save(REFINED_MODEL_PATH)
print(f"⸺ saved to {REFINED_MODEL_PATH}")

# ==================================================
# ② 報酬バランス調整：勝利重視 & 無効行動ペナルティ強化
# ==================================================
TUNED_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_tuned.zip"

# 報酬テーブルを書き換え（環境クラスのクラス変数を直接更新）
tuned_reward = {
    "valid_action_bonus":   0.02,  # 有効手を打ったご褒美
    "win_base":             2.0,   # 勝利基本点
    "win_per_diff":         0.2,   # 勝利時の残枚数差ボーナス
    "lose_base":           -2.0,   # 敗北基本点
    "lose_per_diff":        0.2,   # 敗北時の残枚数差ペナルティ
    "invalid_action_penalty": -0.8 # 無効手ペナルティ
}
PenguinPartyEnv.reward_config = tuned_reward  # ☆ ここで全環境共通に反映

def make_tuned_env():
    """ベクトル環境用のラッパ―関数（毎回新インスタンスを返す）"""
    return PenguinPartyGymEnv()

env_tuned = make_vec_env(make_tuned_env, n_envs=8)  # 並列環境で効率アップ

# ①で得た refined モデルを読み込み、報酬だけ変えて再学習
model_tuned = MaskablePPO.load(
    REFINED_MODEL_PATH,
    env=env_tuned,
    learning_rate=5e-5
)

print("[Step 2] Continue training with tuned reward schedule …")
model_tuned.learn(total_timesteps=10_000, progress_bar=True)
model_tuned.save(TUNED_MODEL_PATH)
print(f"⸺ saved to {TUNED_MODEL_PATH}")

# ==================================================
# ③ 自己対戦リーグ：性格違い BOT とランダム交互対戦
# ==================================================
LEAGUE_MODEL_PATH = FINETUNE_DIR / "ppo_penguin_league.zip"

# 「慎重」モデルと「攻撃的」モデル（entropy 違い）を読み込み
model_safe = MaskablePPO.load("grid_models/ppo_lr5e-05_ent0.001.zip", env=None)
model_aggr = MaskablePPO.load("grid_models/ppo_lr5e-05_ent0.01.zip",  env=None)

class SelfPlayEnv(PenguinPartyGymEnv):
    """自分＋ランダムに選ばれた既存 BOT で 2P 自己対戦する環境"""
    def __init__(self, opponent_model):
        super().__init__()
        self.opponent_model = opponent_model  # 相手エージェント

    def step(self, action_idx):
        # 自分の手番
        obs, reward, done, trunc, info = super().step(action_idx)
        if done:
            return obs, reward, done, trunc, info

        # 相手の行動（マスク付き推論）
        mask = self.action_masks()
        opp_action, _ = self.opponent_model.predict(
            obs, deterministic=True, action_masks=mask
        )
        # 相手の手番を進める
        obs, opp_rew, done, trunc, info = super().step(opp_action)

        # 勝敗を相対評価に変換（相手報酬を反転して自分に加算）
        reward -= opp_rew
        return obs, reward, done, trunc, info

def make_selfplay_env():
    """慎重 or 攻撃的 BOT をランダム選択して返す"""
    opp = random.choice([model_safe, model_aggr])
    return SelfPlayEnv(opp)

vec_sp = make_vec_env(make_selfplay_env, n_envs=8)

league_model = MaskablePPO("MultiInputPolicy", vec_sp, verbose=1, learning_rate=5e-5)
print("[Step 3] Training in self-play league …")
league_model.learn(total_timesteps=10_000, progress_bar=True)
league_model.save(LEAGUE_MODEL_PATH)
print(f"⸺ saved to {LEAGUE_MODEL_PATH}")

# ==================================================
# ④ 局所ランダムサーチ：lr × ent_coef 小規模グリッド探索
# ==================================================
search_space = {
    "lr":  [3e-5, 7e-5],               # 学習率 2 通り
    "ent": [0.002, 0.003, 0.004],      # エントロピー係数 3 通り
}
SAMPLE_STEPS = 10_000  # 各候補の学習ステップ
EVAL_N       = 100     # 評価エピソード数

records = []  # すべての結果を保存して後で DataFrame 化

def quick_eval(m):
    """短時間評価：平均報酬と勝率（%）を返す"""
    env = Monitor(PenguinPartyGymEnv())
    wins, tot_r = 0, []
    for _ in range(EVAL_N):
        o, _ = env.reset()
        done, r = False, 0
        while not done:
            a, _ = m.predict(o, deterministic=True, action_masks=env.env.action_masks())
            o, rew, term, trunc, _ = env.step(a)
            done = term or trunc
            r += rew
        scores = env.env.env.get_scores()  # (self, opp)
        if scores[0] < scores[1]:
            wins += 1
        tot_r.append(r)
    return np.mean(tot_r), wins / EVAL_N * 100  # 平均報酬, 勝率(%)

# ----- 探索ループ -----
for lr, ent in itertools.product(search_space["lr"], search_space["ent"]):
    env = make_vec_env(PenguinPartyGymEnv, n_envs=4)
    m = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=lr,
        ent_coef=ent,
        n_steps=1024,
        batch_size=256,
        verbose=0
    )
    m.learn(total_timesteps=SAMPLE_STEPS)
    mean_r, win = quick_eval(m)

    rand_path = LOCAL_RAND_DIR / f"ppo_lr{lr}_ent{ent}.zip"
    m.save(rand_path)

    records.append((lr, ent, round(mean_r, 3), round(win, 1), str(rand_path)))
    print(f"lr={lr:.0e}, ent={ent:.3f} → reward {mean_r:.2f}, win {win:.1f}%")

# ----- 結果を DataFrame で一覧表示 -----
import pandas as pd
df = (
    pd.DataFrame(records, columns=["lr", "ent", "meanR", "win%", "path"])
      .sort_values("meanR", ascending=False)
)
print("\n========= Local Random Search Result =========")
print(df)
