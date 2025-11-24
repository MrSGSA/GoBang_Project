# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rule import game_rule
from model import game_net
import random
import os
import pickle
from mcts import MCTS  # 👈 新增导入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_human_data(path="human_games.pkl", weight=1):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  ➕ Loaded {len(data)} human samples")
    return data * weight


# ===== 新增：使用 MCTS 的 self-play =====
def self_play_with_mcts(model, env, num_games=100, num_simulations=200):
    """用 MCTS 生成高质量对局"""
    data = []
    model.eval()
    for _ in range(num_games):
        env.reset()
        states, actions, values = [], [], []
        mcts_player = MCTS(model, num_simulations=num_simulations, device=DEVICE)

        while env.winner == 0 and not np.all(env.board):
            states.append(env.board.copy())
            action = mcts_player.run(env)
            actions.append(action)
            env.step(action)

        winner = env.winner
        for i in range(len(states)):
            player = 1 if i % 2 == 0 else -1
            values.append(winner * player)
        data.extend(zip([s.astype(np.float32) for s in states], actions, values))
    return data


def evaluate_model(model, env, num_games=100, use_mcts=False):
    model.eval()
    wins, losses, draws = 0, 0, 0

    with torch.no_grad():
        for _ in range(num_games):
            env.reset()
            if use_mcts:
                # ===== MCTS 模式 =====
                mcts_player = MCTS(model, num_simulations=300, device=DEVICE)
                while env.winner == 0 and not np.all(env.board):
                    action = mcts_player.run(env)
                    env.step(action)
                    # 随机对手
                    if env.winner == 0 and not np.all(env.board):
                        env.step(random.choice(env.get_valid_actions()))
            else:
                # ===== Greedy 模式 =====
                while env.winner == 0 and not np.all(env.board):
                    state_tensor = (
                        torch.tensor(env.board, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    policy_logits, _ = model(state_tensor)
                    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    valid = env.get_valid_actions()
                    mask = np.zeros_like(policy)
                    mask[valid] = 1
                    policy *= mask
                    action = (
                        np.argmax(policy) if policy.sum() > 0 else random.choice(valid)
                    )
                    env.step(action)
                    # 随机对手
                    if env.winner == 0 and not np.all(env.board):
                        env.step(random.choice(env.get_valid_actions()))

            if env.winner == 1:
                wins += 1
            elif env.winner == -1:
                losses += 1
            else:
                draws += 1

    win_rate = wins / num_games
    mode = "MCTS" if use_mcts else "Greedy"
    print(
        f"  📊 Eval ({mode}) vs Random: W{wins} L{losses} D{draws} → Win Rate: {win_rate:.2%}"
    )
    return win_rate


def train(total_epochs=2500, start_epoch=2000):
    env = game_rule()
    model = game_net().to(DEVICE)

    # 🔑 从 epoch 2000 恢复最强模型
    best_ckpt = f"gomoku_model_epoch{start_epoch}.pth"
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        print(f"✅ Resuming from strongest model: {best_ckpt}")
    else:
        raise RuntimeError(f"Model '{best_ckpt}' not found!")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_win_rate = 0.0
    eval_log_path = "win_rate_log.txt"
    if not os.path.exists(eval_log_path):
        with open(eval_log_path, "w") as f:
            f.write("epoch,win_rate\n")

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")

        # ===== 关键：用 MCTS 生成高质量数据 =====
        print("  🌲 Generating MCTS self-play data...")
        data_mcts = self_play_with_mcts(model, env, num_games=80, num_simulations=250)

        # 加入人类数据（防遗忘）
        data_human = load_human_data(weight=2)

        data = data_mcts + data_human
        random.shuffle(data)
        print(f"  Data: {len(data_mcts)} MCTS + {len(data_human) // 2} human(x2)")

        # 训练
        model.train()
        total_loss = 0.0
        batch_size = 64  # MCTS 数据少，减小 batch
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            states = (
                torch.from_numpy(np.stack([d[0] for d in batch]))
                .unsqueeze(1)
                .to(DEVICE)
            )
            actions = torch.tensor([d[1] for d in batch], dtype=torch.long).to(DEVICE)
            values = torch.tensor([[d[2]] for d in batch], dtype=torch.float32).to(
                DEVICE
            )

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)
            loss = policy_criterion(pred_policy, actions) + value_criterion(
                pred_value, values
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(data) // batch_size)
        print(f"  Avg Loss: {avg_loss:.4f}")

        # 每 100 轮评估（同时测 greedy 和 MCTS）
        if (epoch + 1) % 100 == 0:
            win_rate_greedy = evaluate_model(model, env, num_games=50, use_mcts=False)
            win_rate_mcts = evaluate_model(model, env, num_games=50, use_mcts=True)

            # 保存
            ckpt_path = f"gomoku_model_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  📦 Saved {ckpt_path}")

            # 以 MCTS 胜率为准保存最佳模型
            current_best = win_rate_mcts
            if current_best > best_win_rate:
                best_win_rate = current_best
                torch.save(model.state_dict(), "gomoku_best.pth")
                print(f"  🏆 New best model (MCTS) saved! Win rate: {current_best:.2%}")

            with open(eval_log_path, "a") as f:
                f.write(f"{epoch + 1},{win_rate_mcts:.4f}\n")

    torch.save(model.state_dict(), "gomoku_final.pth")
    print(f"\n🎉 Training finished! Best MCTS win rate: {best_win_rate:.2%}")


if __name__ == "__main__":
    train(total_epochs=2500, start_epoch=2000)
