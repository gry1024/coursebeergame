"""On-policy rollout buffer used by PPO, IPPO, and MAPPO."""

import numpy as np
import torch


class RolloutBuffer:
    """Stores one rollout's worth of transitions for a single PPO update.

    Holds Python lists during collection, then converts them to PyTorch tensors
    on demand in ``get(device)``.
    """

    def __init__(self):
        """Initialise empty storage lists."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        """Append one timestep."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        """Reset the buffer for the next rollout."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def get(self, device):
        """Convert stored lists to stacked tensors on the requested device."""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        return states, actions, log_probs, rewards, dones, values

    def __len__(self):
        """Current rollout length."""
        return len(self.states)