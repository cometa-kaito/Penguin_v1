#!/usr/bin/env python3
# ==================================================
#  visualize_game.py
# --------------------------------------------------
# 目的:
#   ・grid_search_results.csv を読み込み、
#     残り手札平均 (remain_avg) が最小＝最も効率的に手札を減らせる
#     モデルを自動選択し、1 ゲームだけ実行＆盤面を可視化する。
#   ・CLI 引数にモデル .zip パスを渡すと手動で指定することも可能。
#
# 使い方:
#   $ python visualize_game.py                       # 自動で最良モデルを選択
#   $ python visualize_game.py path/to/model.zip     # モデルを手動で指定
# ==================================================

import sys
import pandas as pd
from sb3_contrib import MaskablePPO
from penguin_party_gym import PenguinPartyGymEnv

DEFAULT_CSV = "grid_models/grid_search_results.csv"  # グリッドサーチ結果のデフォルト位置


# --------------------------------------------------
#  CSV から最良モデル (.zip パス) を取得
# --------------------------------------------------
def select_best_model(csv_path: str) -> str:
    """
    grid_search_results.csv を読み込み、
    remain_avg が最小 (= 強い) であるモデルのファイルパスを返す。
    """
    df = pd.read_csv(csv_path)
    best_row = df.sort_values("remain_avg").iloc[0]  # 最小値を 1 行抽出
    print(f"✓ remain_avg 最小 = {best_row.remain_avg} のモデルを使用")
    return best_row.model_path


# --------------------------------------------------
#  1 ゲーム可視化 (render)
# --------------------------------------------------
def visualize(model_path: str):
    """
    指定モデルで 1 ゲーム実行し、環境の render() で盤面を逐次表示。
    """
    model = MaskablePPO.load(model_path)  # env=None で OK (後で外部 env 使用)
    env   = PenguinPartyGymEnv()          # 単一環境で可視化

    obs, _ = env.reset()
    done = False
    while not done:
        # 有効手マスクを取得して推論
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        env.render()                     # ここで盤面を標準出力に描画


# --------------------------------------------------
#  エントリーポイント
# --------------------------------------------------
if __name__ == "__main__":
    # 引数があればそれをモデルパスとして使用、無ければ最良モデルを自動選択
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = select_best_model(DEFAULT_CSV)

    print(f"★ 可視化対象モデル: {path}")
    visualize(path)
