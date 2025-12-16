# -*- coding: utf-8 -*-
import numpy as np

# 棋型评分表
SCORE_FIVE = 100000
SCORE_LIVE_4 = 10000
SCORE_DEAD_4 = 1000
SCORE_LIVE_3 = 1000
SCORE_DEAD_3 = 100
SCORE_LIVE_2 = 100

class AI:
    def __init__(self, chessboard_np, my_color=2):
        # 直接接收 numpy 数组 (0:空, 1:黑, 2:白)
        self.board = chessboard_np
        # 【关键修复】自动获取棋盘大小 (19)，不再写死 15
        self.size = chessboard_np.shape[0] 
        self.my_color = my_color          
        self.opp_color = 3 - my_color     

    def get_legal_moves(self):
        """
        获取落子点
        """
        # 获取所有非空点的坐标
        rows, cols = np.nonzero(self.board)
        
        # 【关键修复】如果棋盘是空的，下在正中心 (9,9) 而不是 (7,7)
        if len(rows) == 0:
            center = self.size // 2
            return [(center, center)] 
        
        # 创建搜索区域边界
        # 【关键修复】使用 self.size 限制边界，防止越界或搜索不全
        min_r, max_r = max(0, np.min(rows)-2), min(self.size, np.max(rows)+3)
        min_c, max_c = max(0, np.min(cols)-2), min(self.size, np.max(cols)+3)
        
        moves = set()
        for r, c in zip(rows, cols):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    # 【关键修复】这里用 self.size 判断
                    if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == 0:
                        moves.add((nr, nc))
        
        # 启发式排序：离棋盘中心越近越好
        center = self.size // 2
        sorted_moves = sorted(list(moves), key=lambda p: abs(p[0]-center) + abs(p[1]-center))
        return sorted_moves

    def minimax(self, depth, alpha, beta, is_maximizing):
        # 1. 检查游戏结束或达到深度
        score_eval = self.evaluate_whole_board()
        
        # 如果已经胜利，直接返回巨大的分值，不再搜索
        if abs(score_eval) > SCORE_FIVE * 0.5: 
            return score_eval
        
        if depth == 0:
            return score_eval

        moves = self.get_legal_moves()
        if not moves: return 0

        if is_maximizing:
            max_eval = -float('inf')
            for r, c in moves:
                self.board[r, c] = self.my_color
                eval_val = self.minimax(depth - 1, alpha, beta, False)
                self.board[r, c] = 0 # 回溯
                
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in moves:
                self.board[r, c] = self.opp_color
                eval_val = self.minimax(depth - 1, alpha, beta, True)
                self.board[r, c] = 0 # 回溯
                
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_whole_board(self):
        ai_score = self.evaluate_color_fast(self.my_color)
        opp_score = self.evaluate_color_fast(self.opp_color)
        # 稍微增加防守权重，因为人类很狡猾
        return ai_score - opp_score * 1.5 

    def evaluate_color_fast(self, color):
        score = 0
        lines = []
        
        # 【关键修复】横向遍历使用 self.size
        for r in range(self.size):
            lines.append(self.board[r, :])
        # 【关键修复】纵向遍历使用 self.size
        for c in range(self.size):
            lines.append(self.board[:, c])
            
        # 对角线
        # 19路棋盘对角线更长，范围大约是 -15 到 15
        diag_range = self.size - 5
        for offset in range(-diag_range, diag_range + 1):
            diag = self.board.diagonal(offset)
            if len(diag) >= 5: lines.append(diag)
            anti_diag = np.fliplr(self.board).diagonal(offset)
            if len(anti_diag) >= 5: lines.append(anti_diag)

        # 模式匹配
        for line in lines:
            # 这里的转换稍微耗时，但在Python里正则匹配还算快
            s = "".join(map(str, line))
            
            ptn_5 = str(color) * 5
            ptn_live_4 = "0" + str(color)*4 + "0"
            ptn_dead_4_a = "0" + str(color)*4
            ptn_dead_4_b = str(color)*4 + "0"
            ptn_live_3 = "0" + str(color)*3 + "0"
            # 增加冲三（死三）的判断，防止被偷袭
            ptn_dead_3_a = "0" + str(color)*3
            ptn_dead_3_b = str(color)*3 + "0"
            
            if ptn_5 in s: score += SCORE_FIVE
            elif ptn_live_4 in s: score += SCORE_LIVE_4
            elif ptn_dead_4_a in s or ptn_dead_4_b in s: score += SCORE_DEAD_4
            elif ptn_live_3 in s: score += SCORE_LIVE_3
            elif ptn_dead_3_a in s or ptn_dead_3_b in s: score += SCORE_DEAD_3
            
        return score

    def check_winner(self):
        """
        检查当前棋盘是否有五子连珠。
        返回: 0 - 无胜者; 1 - 黑胜; 2 - 白胜
        """
        board = self.board
        size = self.size

        # 四个方向: 右, 下, 右下, 左下
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(size):
            for c in range(size):
                if board[r, c] == EMPTY:
                    continue
                color = board[r, c]
                for dr, dc in directions:
                    count = 1  # 当前点算一个
                    # 正向检查
                    for i in range(1, 5):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == color:
                            count += 1
                        else:
                            break
                    # 反向检查（避免重复计数，但确保是“连续5”）
                    for i in range(1, 5):
                        nr, nc = r - dr * i, c - dc * i
                        if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == color:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return color
        return 0
