import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, board_size=4):
        super().__init__()
        self.size = board_size
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(32 * (board_size // 2) ** 2, 64)
        self.lin2 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x
    
class ValueNet(nn.Module):
    def __init__(self, board_size=4):
        super().__init__()
        self.size = board_size
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(32 * (board_size // 2) ** 2, 64)
        self.lin2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        return x