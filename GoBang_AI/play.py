import torch
import numpy as np
from rule import game_rule
from model import game_net
import os
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_board(board):
    """在终端打印15x15五子棋棋盘"""
    size = board.shape[0]
    print("\n   ", end="")
    for j in range(size):
        print(f"{j:2d} ", end="")
    print()
    for i in range(size):
        print(f"{i:2d} ", end="")
        for j in range(size):
            if board[i, j] == 1:
                print(" ● ", end="")  # 黑子
            elif board[i, j] == -1:
                print(" ○ ", end="")  # 白子
            else:
                print(" · ", end="")
        print()
    print()


def human_move(env):
    """获取人类玩家输入"""
    while True:
        try:
            inp = input("Your move (row col), e.g. '7 7', or 'q' to quit: ").strip()
            if inp.lower() == "q":
                print("Thanks for playing!")
                exit(0)
            parts = inp.split()
            if len(parts) != 2:
                raise ValueError
            row, col = int(parts[0]), int(parts[1])
            if not (0 <= row < env.size and 0 <= col < env.size):
                print(f"Row and col must be between 0 and {env.size - 1}.")
                continue
            action = row * env.size + col
            if action in env.get_valid_actions():
                return action
            else:
                print("That position is already occupied!")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please enter two numbers like '7 7'.")


def is_winning_move(board, x, y, player, size=15):
    """检查在 (x,y) 落子后是否形成五连（支持斜向）"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1  # 当前子
        # 正方向
        for step in range(1, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                count += 1
            else:
                break
        # 反方向
        for step in range(1, 5):
            nx, ny = x - step * dx, y - step * dy
            if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False


def ai_move(model, env, human_player):
    """AI 下一步：先防胜招，再按策略走"""
    model.eval()
    valid_actions = env.get_valid_actions()

    # 🔒 防守：检查人类下一步是否能赢
    for action in valid_actions:
        temp_board = env.board.copy()
        row, col = action // env.size, action % env.size
        temp_board[row, col] = human_player
        if is_winning_move(temp_board, row, col, human_player, env.size):
            return action

    # 🧠 否则使用模型策略（贪心）
    with torch.no_grad():
        state_tensor = (
            torch.tensor(env.board, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        policy_logits, _ = model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

    # 只考虑合法动作
    mask = np.zeros_like(policy)
    mask[valid_actions] = 1.0
    policy *= mask

    if policy.sum() > 0:
        return int(np.argmax(policy))
    else:
        return valid_actions[0]


def save_human_game(states, actions, winner, human_player, filename="human_games.pkl"):
    """保存整局对局为训练数据"""
    data = []
    for i, (s, a) in enumerate(zip(states, actions)):
        current_player = 1 if i % 2 == 0 else -1
        value = winner * current_player
        data.append((s.astype(np.float32), a, value))

    # 追加到文件
    existing = []
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            existing = pickle.load(f)
    existing.extend(data)
    with open(filename, "wb") as f:
        pickle.dump(existing, f)
    print(f"✅ Game saved! Added {len(data)} samples to '{filename}'.")


def main():
    # 加载模型
    model_path = "gomoku_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' not found!")
        print("Please train the model first or place it in this directory.")
        return

    model = game_net().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Loaded trained model!")

    # 初始化环境
    env = game_rule()
    print("\n🎮 Welcome to Gomoku vs AI!")
    print("● = Black (first), ○ = White")
    print("Board size: 15×15\n")

    # 选择执方
    while True:
        choice = (
            input("Play as Black (●, first) or White (○, second)? [b/w]: ")
            .strip()
            .lower()
        )
        if choice in ["b", "black"]:
            human_player = 1
            print("You are Black. You go first.")
            break
        elif choice in ["w", "white"]:
            human_player = -1
            print("You are White. AI goes first.")
            break
        else:
            print("Please enter 'b' or 'w'.")

    # 开始游戏

    env.reset()
    states, actions = [], []

    while True:
        print_board(env.board)

        # 确定当前该谁走（从 env 获取，不要自己算！）
        current_player = env.current_player

        if current_player == human_player:
            action = human_move(env)
            # 👇 先记录“落子前”的状态（用于训练）
            states.append(env.board.copy())
            actions.append(action)
            _, _, done = env.step(action)  # 落子 + 自动切换玩家
        else:
            action = ai_move(model, env, human_player)
            # 👇 同样记录“落子前”的状态
            states.append(env.board.copy())
            actions.append(action)
            row, col = action // env.size, action % env.size
            _, _, done = env.step(action)
            print(f"AI played at ({row}, {col})")

        if done:
            print_board(env.board)
            winner = env.winner if env.winner is not None else 0
            if winner == human_player:
                print("🎉 You won! Great job!")
            elif winner == -human_player:
                print("💀 AI wins. Better luck next time!")
            else:
                print("🤝 It's a draw!")

            # 保存对局：value = winner * player_at_that_step
            data = []
            for i, (s, a) in enumerate(zip(states, actions)):
                # 第 i 步的玩家：黑先手 → i=0 是黑(1), i=1 是白(-1)...
                player_at_step = 1 if i % 2 == 0 else -1
                value = winner * player_at_step
                data.append((s.astype(np.float32), a, value))

            # 保存
            existing = []
            filename = "human_games.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    existing = pickle.load(f)
            existing.extend(data)
            with open(filename, "wb") as f:
                pickle.dump(existing, f)
            print(f"✅ Game saved! Added {len(data)} samples.")

            break


if __name__ == "__main__":
    main()
