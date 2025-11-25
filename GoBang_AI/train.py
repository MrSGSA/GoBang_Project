# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

from rule import game_rule  # 假设这是你的棋盘逻辑
from model import game_net  # 假设这是你的网络
from mcts import MCTS  # 这是修改后的 MCTS

# --- 全局配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 15
BUFFER_CAPACITY = 30000  # 增大 Buffer，防止遗忘
BATCH_SIZE = 128  # 增大 Batch size，梯度更稳
LR = 2e-4  # 稍微调高一点
L2_REG = 1e-4
CHECKPOINT_FREQ = 1


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, data):
        self.buffer.extend(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def get_equi_data(play_data):
    """
    数据增强：利用旋转和翻转，将 1 条数据扩充为 8 条
    play_data: list of (state, probs, value)
    """
    extended_data = []
    for state, mcts_prob, winner in play_data:
        # state: [15, 15]
        # mcts_prob: [225] -> 还原成 [15, 15] 用于几何变换
        prob_img = mcts_prob.reshape(BOARD_SIZE, BOARD_SIZE)

        for i in [0, 1, 2, 3]:  # 旋转 0, 90, 180, 270 度
            # 1. 旋转
            rot_state = np.rot90(state, i)
            rot_prob = np.rot90(prob_img, i)

            # 添加旋转后的数据
            extended_data.append((rot_state, rot_prob.flatten(), winner))

            # 2. 翻转 (在旋转的基础上进行左右翻转)
            flip_state = np.fliplr(rot_state)
            flip_prob = np.fliplr(rot_prob)

            # 添加翻转后的数据
            extended_data.append((flip_state, flip_prob.flatten(), winner))

    return extended_data


def self_play(model, env, mcts, num_games=1):
    data = []
    model.eval()

    for i in range(num_games):
        env.reset()
        mcts.reset_player()  # 重置 MCTS 树
        states, mcts_probs, current_players = [], [], []

        while True:
            # 获取当前玩家 ID (1 或 -1)
            # 假设 rule.py 中 steps 计数，偶数步是黑(1)，奇数步是白(-1)
            player = 1 if len(env.steps) % 2 == 0 else -1

            # MCTS 搜索
            # temp: 前几步温度高一点，增加探索；后面温度降低，选最好的
            temp = 1.0 if len(env.steps) < 8 else 1e-3
            action, action_probs = mcts.get_action(env, temp=temp, return_prob=1)

            # --- 🔥 关键：存入 canonical state (当前玩家视角) ---
            # 如果当前是白棋(-1)，存进去的盘面要乘以 -1，变成 "1代表己方"
            states.append(env.board * player)
            mcts_probs.append(action_probs)
            current_players.append(player)

            # 执行动作
            env.step(action)

            winner, end = env.has_a_winner()
            if end:
                # winner: 1(黑胜), -1(白胜), 0(平)
                # 为每一步分配 Value
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    for j, p in enumerate(current_players):
                        # 如果 winner == p (这一步的玩家赢了)，则 v = +1
                        # 如果 winner != p (这一步的玩家输了)，则 v = -1
                        winners_z[j] = 1.0 if winner == p else -1.0

                # 打包这一局的数据
                data.extend(get_equi_data(zip(states, mcts_probs, winners_z)))
                break
    return data


def evaluate_network(model, env, mcts, num_games=10):
    """
    评估：当前模型 vs 纯 MCTS (或弱一点的旧模型)
    这里简单起见，做 MCTS vs Random 或者 MCTS (Model) vs MCTS (Weak)
    """
    model.eval()
    mcts_sims = 100  # 评估时不需要太深，速度优先
    wins = 0

    for i in range(num_games):
        env.reset()
        mcts.reset_player()
        mcts.set_simulations(mcts_sims)  # 临时调整模拟次数

        model_player = 1 if i % 2 == 0 else -1  # 轮流执黑

        while True:
            player = 1 if len(env.steps) % 2 == 0 else -1

            if player == model_player:
                # 模型走棋 (低温度，追求最强)
                action = mcts.get_action(env, temp=1e-3)
            else:
                # 对手走棋 (这里用随机作为基准，或者弱 MCTS)
                valid_moves = env.get_valid_actions()
                action = random.choice(valid_moves)

            env.step(action)
            winner, end = env.has_a_winner()
            if end:
                if winner == model_player:
                    wins += 1
                break
    return wins / num_games


def train_cycle(start_epoch=0):
    # 初始化
    env = game_rule()
    model = game_net().to(DEVICE)

    # 加载模型
    if start_epoch > 0:
        model_path = f"gomoku_model_epoch{start_epoch}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded {model_path}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # 初始化 MCTS
    # c_puct: 探索常数，通常 5.0
    mcts = MCTS(model, c_puct=5, num_simulations=400, device=DEVICE)

    for epoch in range(start_epoch, 5000):
        print(f"Epoch {epoch + 1} | Buffer: {len(replay_buffer)}")

        # 1. 动态调整参数
        if epoch < 20:
            games_num, sims, train_steps = 10, 100, 100
        elif epoch < 50:
            games_num, sims, train_steps = 10, 200, 200
        else:
            games_num, sims, train_steps = 10, 400, 300

        # 设置 MCTS 模拟次数
        mcts.set_simulations(sims)

        # 2. 自我对弈收集数据
        new_data = self_play(model, env, mcts, num_games=games_num)
        replay_buffer.push(new_data)

        # 3. 训练
        if len(replay_buffer) > BATCH_SIZE:
            model.train()
            loss_sum = 0
            for _ in range(train_steps):
                batch = replay_buffer.sample(BATCH_SIZE)
                # 解包数据
                state_batch = torch.FloatTensor(np.array([d[0] for d in batch])).to(DEVICE).unsqueeze(
                    1)  # [B, 1, 15, 15]
                mcts_probs_batch = torch.FloatTensor(np.array([d[1] for d in batch])).to(DEVICE)
                winner_batch = torch.FloatTensor(np.array([d[2] for d in batch])).to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                # 前向传播
                log_act_probs, value = model(state_batch)

                # Loss 计算
                # Value Loss (MSE)
                value_loss = nn.MSELoss()(value, winner_batch)
                # Policy Loss (Cross Entropy) - 注意 mcts_probs 是概率，模型输出是 log_softmax
                # 手动计算交叉熵: -sum(target * log_pred)
                policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, dim=1))

                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            print(f"  Loss: {loss_sum / train_steps:.4f}")

        # 4. 保存与评估
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            torch.save(model.state_dict(), f"gomoku_model_epoch{epoch + 1}.pth")

        if (epoch + 1) % 10 == 0:
            win_rate = evaluate_network(model, env, mcts)
            print(f"  📊 Win Rate vs Random: {win_rate:.2%}")


if __name__ == "__main__":
    train_cycle(0)