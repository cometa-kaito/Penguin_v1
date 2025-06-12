import numpy as np
from sb3_contrib import MaskablePPO
from penguin_party_gym import PenguinPartyGymEnv

def play_one_game(model, model_first=True, reward_config=None, verbose=True):
    env = PenguinPartyGymEnv(reward_config=reward_config)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_num = 0
    last_info = None    # <- 追加

    while not done:
        mask = env.unwrapped.action_masks()
        turn = env.env.current_player if hasattr(env.env, "current_player") else env.current_player

        # モデルの手番
        if (turn == 0) == model_first:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            player = "MODEL"
        # ランダムの手番
        else:
            action = int(np.random.choice(np.flatnonzero(mask)))
            player = "RANDOM"

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        total_reward += reward
        step_num += 1
        last_info = info  # <- 追加: 最終infoを保持

        if verbose:
            print(f"Step {step_num}: [{player}] action={action}, reward={reward}, done={done}, info={info}")

        obs = next_obs

    # 勝敗や最終状態出力
    try:
        winner = env.env.outcome(0 if model_first else 1)
    except Exception:
        winner = None
    print(f"=== Game finished ===")
    print(f"Total reward (model perspective): {total_reward}")
    print(f"Winner (for model_first={model_first}): {winner}")
    print(f"Final hand (model): {env.env.hands[0 if model_first else 1]}")
    print(f"Final hand (random): {env.env.hands[1 if model_first else 0]}")
    # --- 追加: ゲーム終了時、両者の最終報酬をinfoから出力 ---
    if last_info is not None and "final_rewards" in last_info:
        print("Final rewards for all players:", last_info["final_rewards"])
    print("-" * 40)
    return total_reward, winner

def main():
    MODEL_PATH = "finetuning/local_rand/ppo_lr7e-05_ent0.003.zip"   # ここを書き換えてOK
    N_GAMES = 1        # ここを好きな回数に変えられます
    REWARD_CONFIG = None  # 任意で報酬設定を渡せる（不要ならNone）

    print(f"Loading model: {MODEL_PATH}")
    model = MaskablePPO.load(MODEL_PATH)

    for i in range(N_GAMES):
        print(f"===== GAME {i+1} =====")
        # 交互に先手後手を変える
        model_first = (i % 2 == 0)
        play_one_game(model, model_first=model_first, reward_config=REWARD_CONFIG, verbose=True)

if __name__ == "__main__":
    main()
