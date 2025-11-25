# model.py
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块：让网络可以做得更深，捕捉更复杂的棋型"""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class game_net(nn.Module):
    def __init__(self, board_size=15, num_channels=128):
        super().__init__()
        self.board_size = board_size

        # 1. 初始卷积块
        self.conv_input = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 2. 残差塔 (建议初期用 4 个，想更强可以用 10-20 个)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(4)
        ])

        # 3. Policy Head (策略头)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, board_size * board_size)
        )

        # 4. Value Head (价值头)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出 [-1, 1] 之间的价值
        )

    def forward(self, x):
        # x: [batch, 1, 15, 15]
        out = self.conv_input(x)

        for block in self.res_blocks:
            out = block(out)

        policy = self.policy_head(out)
        value = self.value_head(out)

        # 🔥🔥 关键修改点 🔥🔥
        # 必须使用 log_softmax，确保输出是"对数概率"
        # 这样配合 train.py 的 NLLLoss (或者手动交叉熵) 以及 mcts.py 的 np.exp 才是对的
        return F.log_softmax(policy, dim=1), value