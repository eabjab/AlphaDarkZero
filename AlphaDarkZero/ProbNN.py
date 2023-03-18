import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ProbabilityNN(nn.Module):
    def __init__(self, num_channels=256, num_residual_layers=40):
        super(ProbabilityNN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(185, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_residual_layers)])
        self.output_layer = nn.Conv2d(num_channels, 6, kernel_size=1, stride=1, padding=0)

    # def forward(self, x):
    #     x = self.input_layer(x)
    #     x = self.residual_layers(x)
    #     x = self.output_layer(x)
    #     x = F.softmax(x, dim=1)
    #     return x

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)

        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, -1)  # Flatten the 8x8 planes
        x = F.softmax(x, dim=-1)  # Apply softmax across the flattened planes
        x = x.view(batch_size, num_channels, height, width)  # Reshape back to the original shape

        return x

