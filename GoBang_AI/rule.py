# rule.py
import numpy as np

class game_rule:
    def __init__(self, size=15):
        self.size = size
        # 🔥 必须加：MCTS 依赖这两个属性，没有会报错
        self.width = size
        self.height = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.winner = 0
        self.steps = []
        self.last_move = -1

    def copy(self):
        new_game = game_rule(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.steps = list(self.steps)
        new_game.last_move = self.last_move
        return new_game

    # 🔥 严重Bug修复：默认参数必须设为 None，否则多次深拷贝会共享内存
    def __deepcopy__(self, memodict=None):
        return self.copy()

    # 🔥 必须加：MCTS 需要调用这个方法判断游戏结束
    def has_a_winner(self):
        if self.winner != 0:
            return self.winner, True
        if len(self.steps) >= self.size * self.size:
            return 0, True
        return 0, False

    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0

    def step(self, action):
        x, y = action // self.size, action % self.size
        if not self.is_valid(x, y):
            raise ValueError(f"Invalid action: {action}")

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

        self.current_player *= -1
        return self.board.copy(), reward, done

    def _check_win(self, x, y):
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                else: break
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    count += 1
                else: break
            if count >= 5: return True
        return False

    def get_valid_actions(self):
        return np.argwhere(self.board.flatten() == 0).flatten().tolist()