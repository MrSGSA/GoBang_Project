# rule.py
import numpy as np

class game_rule:
    def __init__(self, size=15):
        self.size = size
        self.width = size   # ✅ 新增：兼容 mcts.py
        self.height = size  # ✅ 新增：兼容 mcts.py
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1  # Black = 1, White = -1
        self.winner = 0          # 0 = ongoing/draw, 1 = black win, -1 = white win
        self.steps = []          # 记录落子历史 [(x, y), ...]
        self.last_move = -1      # 记录最后一步的 action ID

    def copy(self):
        """
        🔥 关键优化：为 MCTS 提供快速复制
        """
        new_game = game_rule(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.steps = list(self.steps)
        new_game.last_move = self.last_move
        return new_game

    # ✅ 新增：让 copy.deepcopy() 调用你的高效 copy 方法
    def __deepcopy__(self, memodict={}):
        return self.copy()

    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0

    # ✅ 新增：配合 mcts.py 判断游戏结束
    def has_a_winner(self):
        """
        返回: (winner, is_end)
        winner: 1, -1, or 0 (draw)
        is_end: True/False
        """
        if self.winner != 0:
            return self.winner, True
        if len(self.steps) >= self.size * self.size:
            return 0, True
        return 0, False

    def step(self, action):
        """
        执行一步
        action: int (0 ~ 224)
        """
        x, y = action // self.size, action % self.size

        if not self.is_valid(x, y):
            raise ValueError(f"Invalid action: {action} ({x},{y})")

        self.board[x, y] = self.current_player
        self.steps.append((x, y))
        self.last_move = action

        done = False
        reward = 0.0

        if self._check_win(x, y):
            self.winner = self.current_player
            reward = 1.0
            done = True
        elif len(self.steps) >= self.size * self.size:
            self.winner = 0
            reward = 0.0
            done = True

        self.current_player *= -1  # 切换下棋方
        return self.board.copy(), reward, done

    def _check_win(self, x, y):
        """检查 (x,y) 落下后是否形成五连珠"""
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            # 正向
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            # 反向
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return True
        return False

    def get_valid_actions(self):
        """获取所有合法动作的索引列表"""
        return np.argwhere(self.board.flatten() == 0).flatten().tolist()