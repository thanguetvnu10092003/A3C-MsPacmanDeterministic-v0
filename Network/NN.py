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
        
        # Use smaller kernels and strides for 42x42 input
        # This keeps more spatial information
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        # Calculate flattened size based on input size
        # For 42x42: conv1 -> 21x21, conv2 -> 11x11, conv3 -> 11x11 = 64*11*11 = 7744
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            dummy_output = self._forward_conv_simple(dummy_input)
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

    def _forward_conv_simple(self, x):
        """Forward through conv layers without batch norm (for size calculation)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.flatten(x)

    def _forward_conv(self, x):
        """Forward through conv layers with ReLU"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.flatten(x)

    def forward(self, state):
        # Convolutional layers
        x = self._forward_conv(state)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Actor and Critic outputs
        action_values = self.actor(x)
        state_value = self.critic(x)
        
        return action_values, state_value.squeeze(-1)
