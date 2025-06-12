from penguin_party_env import PenguinPartyEnv

def debug_one_game(reward_config=None, verbose=True):
    env = PenguinPartyEnv(reward_config=reward_config)
    obs = env.reset()
    rewards = [0.0 for _ in range(env.num_players)]
    print(f"=== 新規ゲーム開始 ===")
    print(f"初期手札: {[list(h) for h in env.hands]}")

    step = 0
    done = False
    while not done:
        player = env.current_player
        valid_actions = env.get_valid_actions()
        action = valid_actions[0]  # テストなので1つめを機械的に選ぶ

        obs, reward, done, info = env.step(action)
        rewards[player] += reward

        if verbose:
            print(f"Step {step+1} | Player: {player} | Action: {action} | Reward: {reward} | Done: {done} | Info: {info}")
            print(f"手札: {[list(h) for h in env.hands]}")
        step += 1

        # 終局時の全員分reward出力
        if done and "final_rewards" in info:
            print(f"最終報酬: {info['final_rewards']}")
            for i, r in info['final_rewards'].items():
                print(f"Player {i} 最終reward: {r}")

    print(f"ゲーム終了。累積報酬: {rewards}")
    print("-" * 40)

if __name__ == "__main__":
    debug_one_game(verbose=True)
