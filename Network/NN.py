import torch.nn as nn
import torch.nn.functional as F
import torch


class Network(nn.Module):
    """
    Improved A3C Network for Ms. Pac-Man
    Optimized for 4GB GPU with better exploration capabilities
    """

    def __init__(self, action_size, input_channels=4, input_size=42):
        super(Network, self).__init__()
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        # Calculate flattened size based on input size
        # For 42x42: conv1 -> 9x9, conv2 -> 3x3, conv3 -> 1x1 = 64
        # For 84x84: conv1 -> 20x20, conv2 -> 9x9, conv3 -> 7x7 = 3136
        dummy_input = torch.zeros(1, input_channels, input_size, input_size)
        dummy_output = self._forward_conv(dummy_input)
        fc_input_size = dummy_output.shape[1]
        
        # Fully connected layers - moderate size for 4GB GPU
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Actor head (policy)
        self.actor = nn.Linear(256, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(256, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.flatten(x)

    def forward(self, state):
        # Convolutional layers with batch norm and ReLU
        x = self._forward_conv(state)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Actor and Critic outputs
        action_values = self.actor(x)
        state_value = self.critic(x)
        
        return action_values, state_value.squeeze(-1)
