import numpy as np
from ai import AI

def simulate_game():
    print("=== 五子棋 AI 对战模拟器（无摄像头） ===")
    print("黑方（1）先手，白方（2）为 AI\n")

    BOARD_SIZE = 19
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    
    # 黑方先手
    human_color = 1
    ai_color = 2

    move_count = 0

    while True:
        current_color = human_color if move_count % 2 == 0 else ai_color
        is_human_turn = (current_color == human_color)

        # 创建 AI 实例用于评估和胜负判断
        brain = AI(board, my_color=ai_color)

        # 检查是否已有胜者
        winner = brain.check_winner()
        if winner != 0:
            print(f"\n🎉 胜负已分！{'黑方' if winner == 1 else '白方（AI）'} 获胜！")
            break

        # 检查是否平局（棋盘满）
        if np.all(board != 0):
            print("\n🤝 平局！棋盘已满。")
            break

        if is_human_turn:
            print("\n[人类回合] 请输入落子位置（行 列，0~18）：")
            try:
                r, c = map(int, input().strip().split())
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    print("❌ 坐标越界，请重试。")
                    continue
                if board[r, c] != 0:
                    print("❌ 该位置已有棋子，请重试。")
                    continue
                board[r, c] = human_color
                print(f"黑方落子：({r}, {c})")
            except (ValueError, KeyboardInterrupt):
                print("\n👋 用户退出。")
                return
        else:
            print("\n[AI 回合] AI 正在思考...")
            moves = brain.get_legal_moves()
            if not moves:
                print("⚠️ 无合法落子点，游戏结束。")
                break

            best_move = None
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            SEARCH_DEPTH = 2

            for r, c in moves:
                board[r, c] = ai_color
                score = brain.minimax(SEARCH_DEPTH - 1, alpha, beta, False)
                board[r, c] = 0

                if score > best_score:
                    best_score = score
                    best_move = (r, c)
                alpha = max(alpha, score)

            if best_move:
                r, c = best_move
                board[r, c] = ai_color
                print(f"白方（AI）落子：({r}, {c})")
            else:
                # fallback
                r, c = moves[0]
                board[r, c] = ai_color
                print(f"白方（AI）随机落子：({r}, {c})")

        # 打印简易棋盘（只显示最近几步，避免刷屏）
        print(f"当前步数: {move_count + 1}")
        move_count += 1

    # 最终打印小范围棋盘（可选）
    print("\n--- 最终棋盘（中心 7x7）---")
    center = BOARD_SIZE // 2
    half = 3
    sub = board[center-half:center+half+1, center-half:center+half+1]
    for row in sub:
        print(' '.join('.' if x == 0 else ('●' if x == 1 else '○') for x in row))

if __name__ == "__main__":
    simulate_game()
