"""Actor-Critic network shared by PPO and IPPO."""

import torch
import torch.nn as nn


class PPOActorCritic(nn.Module):
    """Two hidden layers followed by an actor head and a value head."""

    def __init__(self, state_size, action_size, hidden_size=256):
        """Initialise shared trunk and the two output heads.

        The actor head is initialised with small weights and zero bias so
        that the initial policy is close to uniform across actions.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.actor.weight.data.mul_(0.01)
        self.actor.bias.data.mul_(0.0)

    def forward(self, x):
        """Return ``(action_probs, state_value)`` for the given state batch."""
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        action_probs = nn.functional.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

    def evaluate(self, states, actions):
        """Compute log-prob, value and entropy for a batch of states and actions."""
        action_probs, state_values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values.squeeze(-1), entropy