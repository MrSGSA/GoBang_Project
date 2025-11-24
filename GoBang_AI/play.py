# play.py
import torch
import numpy as np
from rule import game_rule
from model import game_net
import os
import pickle
from mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_board(board):
    size = board.shape[0]
    print("\n   ", end="")
    for j in range(size):
        print(f"{j:2d} ", end="")
    print()
    for i in range(size):
        print(f"{i:2d} ", end="")
        for j in range(size):
            if board[i, j] == 1:
                print(" ● ", end="")
            elif board[i, j] == -1:
                print(" ○ ", end="")
            else:
                print(" · ", end="")
        print()
    print()


def human_move(env):
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
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for step in range(1, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                count += 1
            else:
                break
        for step in range(1, 5):
            nx, ny = x - step * dx, y - step * dy
            if 0 <= nx < size and 0 <= ny < size and board[nx, ny] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False


def ai_move(model, env, human_player, use_mcts=True):
    if use_mcts:
        mcts = MCTS(model, num_simulations=300, device=DEVICE)
        return mcts.run(env)
    else:
        model.eval()
        valid_actions = env.get_valid_actions()
        for action in valid_actions:
            temp_board = env.board.copy()
            row, col = action // env.size, action % env.size
            temp_board[row, col] = human_player
            if is_winning_move(temp_board, row, col, human_player, env.size):
                return action

        with torch.no_grad():
            state_tensor = (
                torch.tensor(env.board, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )
            policy_logits, _ = model(state_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        mask = np.zeros_like(policy)
        mask[valid_actions] = 1.0
        policy *= mask
        if policy.sum() > 0:
            return int(np.argmax(policy))
        else:
            return valid_actions[0]


def main():
    model_path = "gomoku_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' not found!")
        return

    model = game_net().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Loaded trained model!")

    env = game_rule()
    print("\n🎮 Welcome to Gomoku vs AI!")
    print("● = Black (first), ○ = White")
    print("Board size: 15×15\n")

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

    env.reset()
    states, actions = [], []

    while True:
        print_board(env.board)
        current_player = env.current_player

        if current_player == human_player:
            action = human_move(env)
            states.append(env.board.copy())
            actions.append(action)
            _, _, done = env.step(action)
        else:
            action = ai_move(model, env, human_player, use_mcts=True)
            states.append(env.board.copy())
            actions.append(action)
            row, col = action // env.size, action % env.size
            _, _, done = env.step(action)
            print(f"AI played at ({row}, {col})")

        if done:
            print_board(env.board)
            winner = env.winner  # 已确保为 0/±1
            if winner == human_player:
                print("🎉 You won! Great job!")
            elif winner == -human_player:
                print("💀 AI wins. Better luck next time!")
            else:
                print("🤝 It's a draw!")

            # 保存对局
            data = []
            for i, (s, a) in enumerate(zip(states, actions)):
                player_at_step = 1 if i % 2 == 0 else -1
                value = winner * player_at_step
                data.append((s.astype(np.float32), a, value))

            filename = "human_games.pkl"
            existing = []
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
