import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rule import game_rule
from model import game_net
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def self_play(model, env, num_games=300, temperature=1.0, use_heuristic=True):
    data = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_games):
            env.reset()
            states, actions, values = [], [], []
            while True:
                state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                policy_logits, value = model(state_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

                valid_actions = env.get_valid_actions()

                if use_heuristic and random.random() < 0.15 and len(valid_actions) > 5:
                    neighbor_count = {}
                    for act in valid_actions:
                        x, y = act // env.size, act % env.size
                        count = 0
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < env.size and 0 <= ny < env.size:
                                    if env.board[nx, ny] != 0:
                                        count += 1
                        neighbor_count[act] = count
                    sorted_actions = sorted(neighbor_count.items(), key=lambda x: x[1], reverse=True)
                    top_k = max(1, len(sorted_actions) // 8)
                    candidates = [act for act, _ in sorted_actions[:top_k]]
                    action = random.choice(candidates)
                else:
                    mask = np.zeros(env.size * env.size, dtype=np.float32)
                    mask[valid_actions] = 1.0
                    policy *= mask
                    if policy.sum() == 0:
                        action = random.choice(valid_actions)
                    else:
                        policy = np.power(policy, 1.0 / temperature)
                        policy /= policy.sum()
                        action = np.random.choice(len(policy), p=policy)

                states.append(env.board.astype(np.float32))
                actions.append(action)
                _, reward, done = env.step(action)

                if done:
                    winner = env.winner if env.winner else 0
                    for i in range(len(states)):
                        player = 1 if i % 2 == 0 else -1
                        values.append(winner * player)
                    break

            for s, a, v in zip(states, actions, values):
                data.append((s, a, v))
    return data


def evaluate_model(model, env, num_games=100):
    """评估模型 vs 随机 bot 的胜率"""
    model.eval()
    wins, losses, draws = 0, 0, 0
    with torch.no_grad():
        for _ in range(num_games):
            env.reset()
            turn = 0  # 0: AI, 1: random bot
            while not env.winner and not np.all(env.board):
                if turn == 0:  # AI move
                    state_tensor = torch.tensor(env.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    policy_logits, _ = model(state_tensor)
                    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    valid = env.get_valid_actions()
                    mask = np.zeros_like(policy)
                    mask[valid] = 1
                    policy *= mask
                    if policy.sum() > 0:
                        action = np.argmax(policy)  # 贪心选择
                    else:
                        action = random.choice(valid)
                    env.step(action)
                else:  # random bot
                    valid = env.get_valid_actions()
                    action = random.choice(valid)
                    env.step(action)
                turn = 1 - turn

            if env.winner == 1:
                wins += 1
            elif env.winner == -1:
                losses += 1
            else:
                draws += 1

    win_rate = wins / num_games
    print(f"  📊 Eval vs Random Bot: Wins {wins}, Losses {losses}, Draws {draws} → Win Rate: {win_rate:.2%}")
    return win_rate


def train(total_epochs=2000, start_epoch=900):
    env = game_rule()
    model = game_net().to(DEVICE)

    if os.path.exists("gomoku_final.pth"):
        model.load_state_dict(torch.load("gomoku_final.pth", map_location=DEVICE))
        print(f"✅ Resuming from epoch {start_epoch}")
    else:
        raise RuntimeError("gomoku_final.pth not found!")

    # 加载历史模型池
    model_pool = []
    for ep in range(100, start_epoch + 1, 100):
        path = f"gomoku_model_epoch{ep}.pth"
        if os.path.exists(path):
            m = game_net().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            model_pool.append(m)
            print(f"  ➕ Loaded epoch {ep}")

    print(f"Model pool size: {len(model_pool)}")

    # 初始学习率（前10轮 warm-up）
    base_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # 用于记录胜率
    eval_log_path = "win_rate_log.txt"
    if not os.path.exists(eval_log_path):
        with open(eval_log_path, "w") as f:
            f.write("epoch,win_rate\n")

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")

        # 🔥 Warm-up: 前10轮用更小 lr 防止 loss 飙升
        warmup_steps = 10
        if epoch < start_epoch + warmup_steps:
            lr = base_lr * 0.5  # 前10轮用一半 lr
        else:
            lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 温度（安全！）
        temperature = max(0.15, 1.0 - epoch / (total_epochs * 2))

        # 生成数据
        data_current = self_play(model, env, num_games=400, temperature=temperature)

        # 前5轮禁用 BC，防止策略冲突
        data_bc = []
        if epoch >= start_epoch + 5 and model_pool and random.random() < 0.25:
            expert = random.choice(model_pool)
            bc_temp = max(0.05, temperature * 0.6)
            data_bc = self_play(expert, env, num_games=80, temperature=bc_temp, use_heuristic=False)

        data = data_current + data_bc
        random.shuffle(data)
        print(f"  Data: {len(data_current)} + {len(data_bc)}, Temp: {temperature:.2f}, LR: {lr:.6f}")

        # 训练
        model.train()
        total_p_loss = total_v_loss = 0.0
        batch_size = 128
        num_batches = max(1, len(data) // batch_size)

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            states = torch.from_numpy(np.stack([d[0] for d in batch])).unsqueeze(1).to(DEVICE)
            actions = torch.tensor([d[1] for d in batch], dtype=torch.long).to(DEVICE)
            values = torch.tensor([[d[2]] for d in batch], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)
            p_loss = policy_criterion(pred_policy, actions)
            v_loss = value_criterion(pred_value, values)
            (p_loss + v_loss).backward()
            optimizer.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()

        avg_p_loss = total_p_loss / num_batches
        avg_v_loss = total_v_loss / num_batches
        print(f"  Policy Loss: {avg_p_loss:.4f}, Value Loss: {avg_v_loss:.4f}")

        # 每100轮：保存模型 + 评估胜率 + 更新模型池
        if (epoch + 1) % 100 == 0:
            ckpt_path = f"gomoku_model_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  📦 Saved {ckpt_path}")

            # 更新模型池
            new_model = game_net().to(DEVICE)
            new_model.load_state_dict(model.state_dict())
            new_model.eval()
            model_pool.append(new_model)

            # 📊 评估胜率
            win_rate = evaluate_model(model, env, num_games=100)
            with open(eval_log_path, "a") as f:
                f.write(f"{epoch + 1},{win_rate:.4f}\n")

    torch.save(model.state_dict(), "gomoku_final.pth")
    print("\n🎉 Training finished!")


if __name__ == "__main__":
    train(total_epochs=2000, start_epoch=900)
