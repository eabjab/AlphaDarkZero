import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class RLModel(nn.Module):
    def __init__(self, num_residual_layers=19):
        super(RLModel, self).__init__()
        self.conv_input = nn.Conv2d(17, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        self.residual_layers = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_layers)])
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        out = self.residual_layers(out)
        out = out.view(-1, 256 * 8 * 8)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

