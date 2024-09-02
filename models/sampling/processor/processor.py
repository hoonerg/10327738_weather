# processor.py

import torch
import torch.nn as nn

class ProcessorModel(nn.Module):
    def __init__(self):
        super(ProcessorModel, self).__init__()

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=117, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces to (32x16)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces to (16x8)
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Reduces to (8x4)
        )

        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 4 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, x):
        # Process input
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x