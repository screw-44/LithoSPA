import torch
import torch.nn as nn
# 定义 MLP 模型
class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 预测 3 个索引
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))
