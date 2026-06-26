"""MAPPO networks: per-firm Actor (local) and centralised Critic (global)."""

import torch
import torch.nn as nn


class MAPPOActor(nn.Module):
    """Per-firm Actor that maps a local observation to an action distribution."""

    def __init__(self, state_size, action_size, hidden_size=256):
        """Initialise two hidden layers and a linear actor head.

        The actor head starts near zero so the initial policy is roughly uniform.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.actor.weight.data.mul_(0.01)
        self.actor.bias.data.mul_(0.0)

    def forward(self, x):
        """Return softmax probabilities over the discrete action space."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return nn.functional.softmax(self.actor(x), dim=-1)


class MAPPOCentralizedCritic(nn.Module):
    """Centralised Critic that takes the global state (concatenation of all firms' obs)."""

    def __init__(self, global_state_size, hidden_size=256):
        """Initialise two hidden layers and a scalar value head."""
        super().__init__()
        self.fc1 = nn.Linear(global_state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Return a scalar state-value estimate."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.critic(x)


def build_actor_critic_pair(state_size, action_size, global_state_size, hidden_size=128):
    """Construct one Actor and one Centralised Critic for MAPPO."""
    actor = MAPPOActor(state_size, action_size, hidden_size)
    critic = MAPPOCentralizedCritic(global_state_size, hidden_size)
    return actor, critic