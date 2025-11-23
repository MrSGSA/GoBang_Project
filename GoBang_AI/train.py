import torch
import torch.optim as optim
import numpy as np
from rule import game_rule
from model import game_net
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def self_play(model, env, num_games=100):
    """生成自博弈数据"""
    data = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_games):
            env.reset()
            states, policies, values = [], [], []
            while True:
                state = (
                    torch.tensor(env.board, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                policy, value = model(state)
                policy = policy.cpu().numpy()[0]

                # 掩盖非法动作
                valid_actions = env.get_valid_actions()
                mask = np.zeros(env.size * env.size)
                mask[valid_actions] = 1
                policy *= mask
                if policy.sum() == 0:
                    break
                policy /= policy.sum()

                action = np.random.choice(len(policy), p=policy)
                states.append(env.board.copy())
                policies.append(policy)

                _, reward, done = env.step(action)
                if done:
                    winner = env.winner if env.winner else 0
                    # 胜方视角：+1，负方：-1，平局：0
                    for i, s in enumerate(states):
                        player = 1 if i % 2 == 0 else -1
                        values.append(winner * player)
                    break

            for s, p, v in zip(states, policies, values):
                data.append((s, p, v))
    return data


def train():
    env = game_rule()
    model = game_net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  # 训练50轮
        print(f"Epoch {epoch + 1}/50")

        # 1. 自博弈生成数据
        print("  Generating self-play games...")
        data = self_play(model, env, num_games=200)
        random.shuffle(data)

        # 2. 训练
        model.train()
        total_loss = 0
        batch_size = 64
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            states = (
                torch.tensor([d[0] for d in batch], dtype=torch.float32)
                .unsqueeze(1)
                .to(DEVICE)
            )
            target_policies = torch.tensor(
                [d[1] for d in batch], dtype=torch.float32
            ).to(DEVICE)
            target_values = torch.tensor(
                [[d[2]] for d in batch], dtype=torch.float32
            ).to(DEVICE)

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)

            policy_loss = (
                -(target_policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
            )
            value_loss = torch.nn.functional.mse_loss(pred_value, target_values)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Loss: {total_loss:.4f}")

        # 3. 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"gomoku_model_epoch{epoch + 1}.pth")

    torch.save(model.state_dict(), "gomoku_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train()
