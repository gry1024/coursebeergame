"""Off-policy experience replay buffer for DQN / Double DQN."""

import random
from collections import deque


class ReplayBuffer:
    """Fixed-capacity FIFO buffer storing (state, action, reward, next_state, done) tuples."""

    def __init__(self, capacity):
        """Allocate the deque. Older transitions are discarded when full."""
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Append a single transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Uniformly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        """Current buffer occupancy."""
        return len(self.buffer)