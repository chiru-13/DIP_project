import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class simpleNet(nn.Module):
    def __init__(self, Y=True):
        super(simpleNet, self).__init__()
        # d = 1 for grayscale (Y channel), d = 3 for RGB
        d = 1 if Y else 3
        
        # Define convolutional layers
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Activation function (out-of-place for gradient flow)
        self.relu = nn.ReLU(inplace=False)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x

        # Forward pass through network layers
        x = self.relu(x)
        inputs = self.input(x)
        out = inputs
        
        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        # Uncomment if residual connection before output is desired
        # out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)  # Add residual connection
        return out

