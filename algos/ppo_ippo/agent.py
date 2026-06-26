"""PPO agent and IPPO multi-agent wrapper."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algos.base_agent import BaseAgent
from algos.ppo_ippo.network import PPOActorCritic
from utils.rollout_buffer import RolloutBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNormalizer:
    """Welford running statistics for reward normalisation with optional clipping."""

    def __init__(self, clip=10.0):
        """Initialise counters, mean, and variance to safe defaults."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 1.0
        self.clip = clip

    def update(self, reward):
        """Update running mean and variance with a single sample."""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2

    def normalize(self, reward, update=True):
        """Return the normalised reward, optionally updating the running stats first."""
        if update:
            self.update(reward)
        std = np.sqrt(self.var / self.count) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip)


class PPOAgent(BaseAgent):
    """Single-agent PPO with GAE and clipped surrogate objective."""

    def __init__(self, state_size, action_size, max_order=20, gamma=0.99,
                 lr=3e-4, eps_clip=0.2, K_epochs=3, gae_lambda=0.95,
                 entropy_coef=0.05, min_entropy_coef=0.001, entropy_decay=0.995,
                 value_coef=0.5, **kwargs):
        """Initialise policy + old policy nets, optimiser, rollout buffer, and reward normaliser."""
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.max_order = max_order
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.entropy_decay = entropy_decay
        self.value_coef = value_coef

        self.policy_net = PPOActorCritic(state_size, action_size).to(DEVICE)
        self.old_policy_net = PPOActorCritic(state_size, action_size).to(DEVICE)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = RolloutBuffer()
        self.reward_normalizer = RewardNormalizer()

    def act(self, state, explore=True):
        """Sample (or take argmax of) the current policy and return action, log-prob, value."""
        state_tensor = torch.FloatTensor(np.asarray(state).flatten()).to(DEVICE)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value = self.old_policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            if explore:
                action_index = dist.sample()
            else:
                action_index = torch.argmax(action_probs, dim=-1)
            log_prob = dist.log_prob(action_index)
        action = int(action_index.item())
        action = min(action, self.max_order)
        return action, int(action_index.item()), float(log_prob.item()), float(state_value.item())

    def step(self, state, action_index, log_prob, reward, next_state, done, value=0.0):
        """Append one timestep to the rollout buffer. The actual PPO update happens later."""
        normalized_reward = self.reward_normalizer.normalize(reward, update=True)
        self.memory.push(state, action_index, log_prob, normalized_reward, done, value)

    def update(self, next_state=None, done=True):
        """Run ``K_epochs`` of clipped PPO updates on the buffered rollout."""
        if len(self.memory) == 0:
            return
        states, actions, old_log_probs, rewards, dones, values = self.memory.get(DEVICE)

        # Bootstrap value at the end of the rollout.
        next_value = 0.0
        if next_state is not None and not done:
            next_state_tensor = torch.FloatTensor(np.asarray(next_state).flatten()).to(DEVICE)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
            with torch.no_grad():
                _, next_value_tensor = self.old_policy_net(next_state_tensor)
                next_value = next_value_tensor.item()

        returns, advantages = self._compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.K_epochs):
            log_probs, state_values, entropy = self.policy_net.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns_normalized)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Sync old policy with the updated policy.
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.memory.clear()
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)

    def _compute_gae(self, rewards, values, dones, next_value):
        """Generalised advantage estimation, processed backwards from the rollout's last step."""
        rewards_np = rewards.cpu().numpy()
        values_np = values.cpu().numpy()
        dones_np = dones.cpu().numpy()
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards_np))):
            next_v = values_np[t + 1] if t + 1 < len(values_np) else next_value
            delta = rewards_np[t] + self.gamma * next_v * (1 - dones_np[t]) - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_np[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(np.array(advantages)).to(DEVICE)
        returns = advantages + values
        return returns, advantages

    def save(self, path):
        """Persist the policy network."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Load the policy network and copy it into the old-policy buffer."""
        if not os.path.isfile(path):
            return False
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        return True


class IPPOAgent(BaseAgent):
    """N independent PPO agents, one per controlled firm, with no inter-agent communication."""

    def __init__(self, num_agents, state_size, action_size, **kwargs):
        """Create ``num_agents`` independent ``PPOAgent`` instances."""
        super().__init__()
        self.num_agents = num_agents
        self.agents = [PPOAgent(state_size, action_size, **kwargs) for _ in range(num_agents)]

    def act(self, states, explore=True):
        """Have each agent pick an action given its local observation."""
        actions = []
        infos = []
        for i, agent in enumerate(self.agents):
            action, a_idx, log_prob, value = agent.act(states[i], explore=explore)
            actions.append(action)
            infos.append({"action_index": a_idx, "log_prob": log_prob, "value": value})
        return actions, infos

    def step(self, states, infos, rewards, next_states, dones):
        """Push each agent's transition into its own buffer."""
        for i, agent in enumerate(self.agents):
            agent.step(
                states[i],
                infos[i]["action_index"],
                infos[i]["log_prob"],
                rewards[i],
                next_states[i],
                dones,
                value=infos[i]["value"],
            )

    def update(self, next_states=None, done=True):
        """Update each agent independently."""
        for i, agent in enumerate(self.agents):
            ns = next_states[i] if next_states is not None else None
            agent.update(ns, done)

    def save(self, path_prefix):
        """Save each agent under ``{path_prefix}_agent_{i}.pth``."""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{i}.pth")

    def load(self, path_prefix):
        """Load each agent from ``{path_prefix}_agent_{i}.pth``."""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{i}.pth")
        return True