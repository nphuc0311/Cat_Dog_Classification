import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=1):  # ch_in, ch_out, kernel, stride, padding
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 128, 3, 1, 1)
        self.conv3 = Conv(128, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        for _ in range(3):
          x = self.pool(self.conv3(x))

        # Flatten before fully connected layers
        x = x.view(-1, 128 * 7 * 7)

        # Apply dropout after flattening and before the fully connected layers
        x = self.dropout(x)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
