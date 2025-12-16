import time
import numpy as np
from detection import GobangVision
from ai import AI  # 确保已包含 check_winner

def find_best_move(current_board, ai_color=2):
    brain = AI(current_board, my_color=ai_color)
    moves = brain.get_legal_moves()
    print(f"AI 正在思考... (候选点数量: {len(moves)})")

    best_score = -float('inf')
    best_move = None
    SEARCH_DEPTH = 2 
    alpha = -float('inf')
    beta = float('inf')
    
    start_time = time.time()
    for r, c in moves:
        brain.board[r, c] = ai_color
        score = brain.minimax(SEARCH_DEPTH - 1, alpha, beta, False)
        brain.board[r, c] = 0
        
        if score > best_score:
            best_score = score
            best_move = (r, c)
        alpha = max(alpha, score)
        
    end_time = time.time()
    print(f"AI 思考耗时: {end_time - start_time:.2f}秒")
    return best_move

def run_demo():
    vision = GobangVision(camera_id=0, rotate_image=1) 
    vision.start()
    
    print("=== 视觉五子棋 AI 启动 ===")
    print("AI 执白 (2), 人类执黑 (1)")
    print("按 'q' 退出，按 'r' 重置棋盘\n")
    
    last_black_count = 0
    ai_color = 2
    game_over = False

    try:
        while True:
            if game_over:
                print("游戏已结束。按任意键退出...")
                time.sleep(5)
                break

            board = vision.get_current_board()
            curr_black = np.sum(board == 1)
            curr_white = np.sum(board == 2)

            # ====== 【新增】胜负检测 ======
            brain = AI(board, my_color=ai_color)
            winner = brain.check_winner()
            if winner != 0:
                print("\n" + "="*40)
                if winner == 1:
                    print("🎉 人类（黑方）获胜！")
                else:
                    print("🤖 AI（白方）获胜！")
                print("="*40)
                game_over = True
                continue
            # ==============================

            # 轮到 AI 落子（黑子刚下完）
            if curr_black > last_black_count and curr_black > curr_white:
                print(f"\n[检测] 轮到 AI 落子 (黑:{curr_black}, 白:{curr_white})")
                time.sleep(1.0)
                board_stable = vision.get_current_board()

                # 再次确认轮到 AI
                if np.sum(board_stable == 1) > np.sum(board_stable == 2):
                    move = find_best_move(board_stable, ai_color)
                    if move:
                        print(f"=============================")
                        print(f"!!! AI 建议坐标: 行 {move[0]}, 列 {move[1]} !!!")
                        print(f"=============================")
                        vision.set_ai_hint(move)
                        last_black_count = curr_black
                    else:
                        print("AI 无法决策")

            # 如果人类已落白子（错误操作），或 AI 落子后人类跟进了，清除提示
            if np.sum(board == 2) > curr_white:
                vision.set_ai_hint(None)

            # 更新黑子计数（防抖）
            if curr_black == np.sum(board == 1):
                last_black_count = curr_black
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        print("\n系统已退出")

if __name__ == "__main__":
    run_demo()
