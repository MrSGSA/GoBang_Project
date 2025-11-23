import torch
import torch.nn as nn
import torch.nn.functional as F


class game_net(nn.Module):
    def __init__(self, board_size=15, num_channels=64):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.policy_head = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(num_channels, 1), nn.Tanh()
        )

    def forward(self, x):
        # x: [B, 1, 15, 15]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy = self.policy_head(x)  # [B, 1, 15, 15]
        policy = policy.view(-1, self.board_size * self.board_size)  # [B, 225]
        value = self.value_head(x)  # [B, 1]
        return F.softmax(policy, dim=1), value
