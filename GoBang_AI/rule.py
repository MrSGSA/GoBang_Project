import numpy as np


class game_rule:
    def __init__(self, size=15):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.winner = 0  # 0=未结束/平局, 1=黑胜, -1=白胜

    def is_valid(self, x, y):  # 修复拼写：vaild → valid
        return (
            0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0
        )  # 修复边界：<= → <

    def step(self, action):
        x, y = action // self.size, action % self.size
        if not self.is_valid(x, y):
            return self.board.copy(), -10, True

        self.board[x, y] = self.current_player
        if self._check_win(x, y):
            self.winner = self.current_player
            reward = 1.0
            done = True
        elif np.all(self.board != 0):
            reward = 0.0  # 平局
            done = True
        else:
            reward = 0.0
            done = False

        self.current_player *= -1  # 1 → -1 → 1 ...
        return self.board.copy(), reward, done

    def _check_win(self, x, y):
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[nx, ny] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[nx, ny] == player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_valid_actions(self):
        return [
            i
            for i in range(self.size * self.size)
            if self.board[i // self.size, i % self.size] == 0
        ]
