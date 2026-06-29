"""Small CNN for Fashion-MNIST (28x28 grayscale, 10 classes)."""

import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # padding=1 with a 3x3 kernel preserves spatial size; only pooling shrinks it.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two poolings take 28 -> 14 -> 7, so the flattened length is 64 * 7 * 7.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits; softmax is applied inside CrossEntropyLoss