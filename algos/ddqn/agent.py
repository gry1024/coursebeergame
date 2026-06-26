"""Double DQN agent."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algos.base_agent import BaseAgent
from algos.dqn.network import QNetwork
from utils.replay_buffer import ReplayBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDQNAgent(BaseAgent):
    """Double DQN: decouples action selection (online net) from value estimation (target net)."""

    def __init__(self, state_size, action_size, max_order=20, buffer_size=10000,
                 batch_size=64, gamma=0.99, learning_rate=1e-3, tau=1e-3,
                 update_every=4, **kwargs):
        """Same hyper-parameters as DQN; only the target computation differs."""
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        self.learning_step = 0

        self.q_network = QNetwork(state_size, action_size).to(DEVICE)
        self.target_network = QNetwork(state_size, action_size).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, epsilon=0.0):
        """Epsilon-greedy action selection identical to DQN."""
        state_tensor = torch.from_numpy(np.asarray(state).flatten()).float().unsqueeze(0).to(DEVICE)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        return random.randint(0, self.max_order)

    def step(self, state, action, reward, next_state, done):
        """Store transition and trigger periodic learning."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        """One Double-DQN update.

        The online net selects the next action; the target net evaluates it.
        This reduces the over-estimation bias of standard DQN.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack([np.asarray(s).flatten() for s in states])).float().to(DEVICE)
        actions = torch.LongTensor([int(a) for a in actions]).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor([[float(r)] for r in rewards]).to(DEVICE)
        next_states = torch.from_numpy(np.vstack([np.asarray(ns).flatten() for ns in next_states])).float().to(DEVICE)
        dones = torch.FloatTensor([[float(d)] for d in dones]).to(DEVICE)

        # Online net picks the greedy next action.
        next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        # Target net evaluates that action.
        q_targets_next = self.target_network(next_states).gather(1, next_actions)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.q_network(states).gather(1, actions)
        loss = nn.MSELoss()(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()
        return loss.item()

    def soft_update(self):
        """Polyak-average target network parameters toward the online network."""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self, *args, **kwargs):
        """No-op: DDQN learns inside ``step``."""
        pass

    def save(self, path):
        """Persist both networks and the optimiser."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load both networks and the optimiser. Returns False if file is missing."""
        if not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return True