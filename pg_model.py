import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, board_size=4, inflation=16):
        super().__init__()
        self.size = board_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, inflation, kernel_size=3, padding=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(inflation * board_size ** 2, 16)
        self.lin2 = nn.Linear(16, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.lin2(x)
        x = self.softmax(x)
        return x