import time
import numpy as np
from detection import GobangVision
from ai import AI  # 导入上面修改后的 AI 类

def find_best_move(current_board, ai_color=2):
    """
    AI 决策入口
    """
    # 1. 初始化 AI
    brain = AI(current_board, my_color=ai_color)
    
    # 2. 获取候选点 (只在有子的地方周围搜)
    moves = brain.get_legal_moves()
    print(f"AI 正在思考... (候选点数量: {len(moves)})")
    
    best_score = -float('inf')
    best_move = None
    
    # 3. 根节点搜索 (第一层循环写在外面方便打印进度)
    # 如果电脑太慢，把 depth 改为 1；如果太快但下得烂，改为 3
    SEARCH_DEPTH = 2 
    
    alpha = -float('inf')
    beta = float('inf')
    
    start_time = time.time()
    
    for r, c in moves:
        # 尝试落子
        brain.board[r, c] = ai_color
        
        # 递归计算分数 (轮到对手下，Minimize)
        score = brain.minimax(SEARCH_DEPTH - 1, alpha, beta, False)
        
        # 回溯
        brain.board[r, c] = 0
        
        # 更新最佳
        if score > best_score:
            best_score = score
            best_move = (r, c)
            # print(f"  更新最佳: {best_move} 分数: {best_score}")
            
        # Alpha 更新 (根节点是 Maximizing)
        alpha = max(alpha, score)
        
    end_time = time.time()
    print(f"AI 思考耗时: {end_time - start_time:.2f}秒")
    
    return best_move

# ... 你的 AI 代码 ...

def run_demo():
    # 记得根据你的实际情况设置 rotate_image
    vision = GobangVision(camera_id=0, rotate_image=1) 
    vision.start()
    
    print("=== 视觉五子棋 AI 启动 ===")
    print("AI 执白 (2), 人类执黑 (1)")
    
    last_black_count = 0
    ai_color = 2
    
    try:
        while True:
            board = vision.get_current_board()
            curr_black = np.sum(board == 1)
            curr_white = np.sum(board == 2)
            
            # 轮到 AI 落子
            if curr_black > last_black_count and curr_black > curr_white:
                print(f"\n[检测] 轮到 AI 落子 (黑:{curr_black}, 白:{curr_white})")
                
                time.sleep(1.0)
                board_stable = vision.get_current_board()
                
                if np.sum(board_stable == 1) > np.sum(board_stable == 2):
                    move = find_best_move(board_stable, ai_color)
                    
                    if move:
                        print(f"=============================")
                        print(f"!!! AI 建议坐标: 行 {move[0]}, 列 {move[1]} !!!")
                        print(f"=============================")
                        
                        # 【关键】把 AI 的建议传回给视觉系统
                        # 屏幕上会出现一个紫色的点，请把白子下在紫点上！
                        vision.set_ai_hint(move)
                        
                        last_black_count = curr_black
                    else:
                        print("AI 无法决策")
            
            # 检测玩家是否已经下了白子 (根据数量判断)
            # 如果白子数量增加了，说明玩家已经跟进，清除屏幕上的提示
            if np.sum(board == 2) > curr_white:
                 vision.set_ai_hint(None)

            # 更新黑子计数
            if curr_black == np.sum(board == 1):
                last_black_count = curr_black
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        vision.stop()
        print("\n退出系统")

if __name__ == "__main__":
    run_demo()