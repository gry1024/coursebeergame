"""MLP Q-network shared by DQN and Double DQN."""

import torch.nn as nn


class QNetwork(nn.Module):
    """Three-layer fully connected network mapping state to per-action Q values."""

    def __init__(self, state_size, action_size, hidden_size=64):
        """Initialise two hidden layers and a linear output head of size ``action_size``."""
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """Run a forward pass and return Q-values for each action."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)