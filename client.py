# client.py ────────────────────────────────────────
# --------------------------------------------------
#  役割
#    - サーバとソケット通信し、Penguin Party プロトコルに従って
#      * 名前登録
#      * 評価値送信
#      * モデルが選んだ手を送信
#    - 各ターンで「合法手の確率分布」をコンソールに表示する
# --------------------------------------------------

import socket, json, time, sys
import ai   # ai.load_model(), Evaluation(), ModelPlay()

# ------------------------------------------------------------------
#  送信ユーティリティ
# ------------------------------------------------------------------
def send_json(sock: socket.socket, obj: dict):
    """
    obj を JSON 文字列にして改行付きで送信。
    サーバ側は行末 '\n' を区切りにメッセージを判断している。
    """
    sock.send((json.dumps(obj, separators=(',', ':')) + '\n').encode('utf-8'))

# ------------------------------------------------------------------
#  合法手分布の可視化
# ------------------------------------------------------------------
def print_distribution(dist, chosen_pos, chosen_card):
    """
    Parameters
    ----------
    dist : List[(pos, card, prob)]
        ai.get_action_distribution() の返値
    chosen_pos / chosen_card : 今回実際に採択された手
    """
    print("\n--- 全合法手と行動確率---")
    for pos, card, prob in sorted(dist, key=lambda x: -x[2])[:15]:
        mark = " <= chosen" if (pos == chosen_pos and card == chosen_card) else ""
        print(f"{pos or 'skip':>6}  {card or '---':>6}  {prob:.3f}{mark}")
    print("----------------------------------\n")

# ------------------------------------------------------------------
#  受信メッセージのハンドラ
# ------------------------------------------------------------------
def handle(sock, msg, board, hand, pending_id):
    """
    サーバから 1 メッセージ受信するたびに呼ばれる。
    状態を更新し、必要に応じて応答 JSON を送信する。

    Returns
    -------
    (game_over, board, hand, pending_id)
    """
    typ = msg.get("Type")

    # --- 接続開始：プレイヤー名送信 ---
    if typ == "ConnectionStart":
        send_json(sock, {
            "Type": "PlayerName",
            "Name": "TajimaLab",
            "From": "Client",
            "To": "Server"
        })

    # --- 名前が受理されゲーム開始 ---
    elif typ == "NameReceived":
        print("Start")

    # --- 盤面情報更新 ---
    elif typ == "BoardInfo":
        board = msg["Cardlist"]

    # --- 手札情報更新 ---
    elif typ == "HandInfo":
        hand = msg["Cardlist"]

    # --- サーバから評価値要求 ---
    elif typ == "DoPlay":
        ev_dict = {c: f"{v:.2f}" for c, v in ai.Evaluation(hand)}
        send_json(sock, {
            "Type": "Evaluation",
            **ev_dict,
            "From": "Client",
            "To": "Server"
        })

    # --- 着手許可：MessageID を保持し実際の手を送る ---
    elif typ == "Accept":
        pending_id = msg["MessageID"]

        # モデルで手を決定 & 全合法手分布取得
        pos, card, dist = ai.ModelPlay(board, hand)
        print_distribution(dist, pos, card)

        send_json(sock, {
            "Type": "Play",
            "Position": pos or "",
            "Card": card or "",
            "MessageID": pending_id,
            "From": "Client",
            "To": "Server"
        })
        print(f"AI → {pos} {card}")

    # --- ゲーム終了 ---
    elif typ == "GameEnd":
        return True, board, hand, pending_id

    return False, board, hand, pending_id

# ------------------------------------------------------------------
#  メイン処理
# ------------------------------------------------------------------
if __name__ == "__main__":
    # ---- モデルのロード (CLI 引数で差し替え可能) ----
    if len(sys.argv) >= 2:
        ai.load_model(sys.argv[1])
        print("◆ 使用モデル:", sys.argv[1])
    else:
        print("◆ 使用モデル:", ai.DEFAULT_MODEL_PATH)

    # ---- サーバへ接続 ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 12052))
    print("サーバに接続しました (IP=localhost, Port=12052)")

    # ---- 通信ループ ----
    buffer = ""
    board, hand, pending_id = {}, [], None
    game_over = False

    while not game_over:
        # 受信バッファに追記
        buffer += sock.recv(4096).decode()

        # 行単位でメッセージを取り出す
        while '\n' in buffer:
            raw_line, buffer = buffer.split('\n', 1)
            if not raw_line.strip():
                continue
            msg_json = json.loads(raw_line)

            game_over, board, hand, pending_id = handle(
                sock, msg_json, board, hand, pending_id
            )

        time.sleep(0)  # CPU 負荷を下げる軽いスリープ

    # ---- 終了処理 ----
    sock.close()
    print("接続を終了しました。")
