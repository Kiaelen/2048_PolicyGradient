import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from env import NUM_CHANNEL

class Encoder(nn.Module):
    def __init__(self, board_size=4):
        super().__init__()
        self.res1 = nn.Conv2d(NUM_CHANNEL, 128, kernel_size=1)
        self.conv1 = nn.Conv2d(NUM_CHANNEL, 128, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(16, 128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(16, 128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.group_norm3 = nn.GroupNorm(16, 128)
        
        self.flatten = nn.Flatten()
        
        self.res2 = nn.Linear(128 * (board_size ** 2), 128)
        self.lin1 = nn.Linear(128 * (board_size ** 2), 128)
        self.lin2 = nn.Linear(128, 128)
    
    def forward(self, x):
        x_res = self.res1(x)
        x = self.conv1(x)
        x = self.group_norm1(F.gelu(x))
        x = self.conv2(x)
        x = self.group_norm2(F.gelu(x))
        x = self.conv3(x)
        x = self.group_norm3(F.gelu(x))
        x += x_res
        
        x = self.flatten(x)
        
        x_res = self.res2(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x += x_res
        
        return x
    
class PolicyNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.res = nn.Linear(128, 4)
        self.dropout = nn.Dropout(p=0.2)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x_res = self.res(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x += x_res
        x = self.softmax(x)
        return x
    
class ValueNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.res = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x_res = self.res(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x += x_res
        return x
    
class OriginalPolicyNet(nn.Module):
    def __init__(self, board_size=4):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 128, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(16, 128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(16, 128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.group_norm3 = nn.GroupNorm(16, 128)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(128 * (board_size // 2) ** 2, 128)
        self.lin2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.group_norm1(F.gelu(x))
        x = self.conv2(x)
        x = self.group_norm1(F.gelu(x))
        x = self.conv3(x)
        x = self.group_norm1(F.gelu(x))
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x