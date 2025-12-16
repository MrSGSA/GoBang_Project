import cv2
import numpy as np
import time
from ai import AI

BOARD_SIZE = 19
EMPTY, BLACK, WHITE = 0, 1, 2

def create_virtual_board_image(board):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (130, 205, 238)  # 浅蓝背景
    step = 500 // (BOARD_SIZE + 1)

    # 画线
    for i in range(BOARD_SIZE):
        pos = step * (i + 1)
        cv2.line(img, (pos, step), (pos, 500 - step), (0, 0, 0), 1)
        cv2.line(img, (step, pos), (500 - step, pos), (0, 0, 0), 1)

    # 星位
    stars = [3, 9, 15]
    for r in stars:
        for c in stars:
            cv2.circle(img, (step * (c + 1), step * (r + 1)), 3, (0, 0, 0), -1)

    # 棋子
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            state = board[r, c]
            cx = step * (c + 1)
            cy = step * (r + 1)
            if state == BLACK:
                cv2.circle(img, (cx, cy), 11, (10, 10, 10), -1)
            elif state == WHITE:
                cv2.circle(img, (cx, cy), 11, (240, 240, 240), -1)
                cv2.circle(img, (cx, cy), 11, (100, 100, 100), 1)
    return img

def get_black_move(board):
    """黑方：轻量级 AI（评估打分选最佳）"""
    brain = AI(board, my_color=BLACK)
    moves = brain.get_legal_moves()
    if not moves:
        return None
    best_move = moves[0]
    best_score = -1
    for r, c in moves:
        board[r, c] = BLACK
        score = brain.evaluate_color_fast(BLACK)
        board[r, c] = EMPTY
        if score > best_score:
            best_score = score
            best_move = (r, c)
    return best_move

def get_white_move(board):
    """白方：你的主 AI（depth=2）"""
    brain = AI(board, my_color=WHITE)
    moves = brain.get_legal_moves()
    if not moves:
        return None

    best_score = -float('inf')
    best_move = moves[0]
    SEARCH_DEPTH = 2
    alpha = -float('inf')
    beta = float('inf')

    for r, c in moves:
        brain.board[r, c] = WHITE
        score = brain.minimax(SEARCH_DEPTH - 1, alpha, beta, False)
        brain.board[r, c] = EMPTY

        if score > best_score:
            best_score = score
            best_move = (r, c)
        alpha = max(alpha, score)

    return best_move

def simulate_auto():
    print("=== 全自动五子棋对战（仅虚拟棋盘 + 胜负判断） ===")
    print("黑方（1）: 轻量AI\n白方（2）: 主AI（depth=2）\n")

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    move_count = 0

    # 创建窗口
    cv2.namedWindow("Virtual Board", cv2.WINDOW_AUTOSIZE)
    print("初始化窗口...")
    time.sleep(1.5)  # 等待窗口加载

    try:
        while True:
            # 检查胜负
            checker = AI(board, my_color=WHITE)
            winner = checker.check_winner()
            if winner != 0:
                print("\n" + "="*50)
                print(f"🎉 {'黑方' if winner == 1 else '白方（AI）'} 获胜！总步数: {move_count}")
                print("="*50)
                break

            if np.all(board != EMPTY):
                print("\n🤝 平局！棋盘已满。")
                break

            is_black_turn = (move_count % 2 == 0)

            if is_black_turn:
                move = get_black_move(board)
                if move:
                    r, c = move
                    board[r, c] = BLACK
                    print(f"[黑方落子] ({r}, {c})")
                else:
                    print("黑方无合法落子")
                    break
            else:
                move = get_white_move(board)
                if move:
                    r, c = move
                    print(f"=============================")
                    print(f"!!! AI 建议坐标: 行 {r}, 列 {c} !!!")
                    print(f"=============================")
                    board[r, c] = WHITE
                else:
                    print("白方 AI 无法决策")
                    break

            # 更新虚拟棋盘显示
            virtual_img = create_virtual_board_image(board)
            cv2.imshow("Virtual Board", virtual_img)
            cv2.waitKey(1)  # 必须调用才能刷新

            move_count += 1
            time.sleep(2.5)  # 👈 关键：加长间隔，避免卡顿刷屏

        # 游戏结束后再显示几秒
        time.sleep(3)
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n用户中断")

if __name__ == "__main__":
    simulate_auto()
