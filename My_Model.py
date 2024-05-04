import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(644096, 64)  # Correct input size
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # N, 3, 32, 32
        x = nn.ReLU()(self.conv1(x))   # 314×154×32
        x = self.pool(x)            # 157×77×32
        x = nn.ReLU()(self.conv2(x))   # 153×73×128
        x = self.pool(x)            # 76×36×128
        x = nn.ReLU()(self.conv3(x))   # 74×34×256
        x = nn.Flatten()(x)     # 644096x1
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x