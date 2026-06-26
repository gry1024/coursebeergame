"""MAPPO single-agent and multi-agent wrappers (CTDE)."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algos.base_agent import BaseAgent
from algos.mappo.network import build_actor_critic_pair
from utils.rollout_buffer import RolloutBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNormalizer:
    """Welford running statistics with optional clipping; identical to the PPO version."""

    def __init__(self, clip=10.0):
        """Initialise counters, mean, and variance."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 1.0
        self.clip = clip

    def update(self, reward):
        """Update running mean and variance with one sample."""
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
        return np.clip((reward - self.mean) / std, -self.clip, self.clip)


class MAPPOAgent(BaseAgent):
    """Single MAPPO firm: local Actor + centralised Critic (CTDE)."""

    def __init__(self, state_size, action_size, global_state_size, max_order=20,
                 gamma=0.99, lr=3e-4, eps_clip=0.2, K_epochs=3, gae_lambda=0.95,
                 entropy_coef=0.05, min_entropy_coef=0.001, entropy_decay=0.995,
                 value_coef=0.5, **kwargs):
        """Initialise Actor, Critic, their old-policy copies, optimiser, and rollout buffer.

        The Critic sees the global state (size = ``num_agents * local_obs_dim``);
        the Actor still only sees its own local observation.
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.global_state_size = global_state_size
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

        self.actor, self.critic = build_actor_critic_pair(
            state_size, action_size, global_state_size
        )
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.old_actor, self.old_critic = build_actor_critic_pair(
            state_size, action_size, global_state_size
        )
        self.old_actor.to(DEVICE)
        self.old_critic.to(DEVICE)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )
        self.memory = RolloutBuffer()
        self.reward_normalizer = RewardNormalizer()
        self.global_states = []  # parallel list of global states, one per timestep

    def act(self, local_state, global_state, explore=True):
        """Sample an action from the Actor; read the value from the centralised Critic."""
        local_tensor = torch.FloatTensor(np.asarray(local_state).flatten()).to(DEVICE)
        if local_tensor.dim() == 1:
            local_tensor = local_tensor.unsqueeze(0)
        global_tensor = torch.FloatTensor(np.asarray(global_state).flatten()).to(DEVICE)
        if global_tensor.dim() == 1:
            global_tensor = global_tensor.unsqueeze(0)
        with torch.no_grad():
            action_probs = self.old_actor(local_tensor)
            state_value = self.old_critic(global_tensor)
            dist = torch.distributions.Categorical(action_probs)
            if explore:
                action_index = dist.sample()
            else:
                action_index = torch.argmax(action_probs, dim=-1)
            log_prob = dist.log_prob(action_index)
        action = int(action_index.item())
        action = min(action, self.max_order)
        return action, int(action_index.item()), float(log_prob.item()), float(state_value.item())

    def step(self, local_state, global_state, action_index, log_prob, reward, next_local_state,
             next_global_state, done, value=0.0):
        """Buffer one timestep together with the global state at this step."""
        normalized_reward = self.reward_normalizer.normalize(reward, update=True)
        self.memory.push(local_state, action_index, log_prob, normalized_reward, done, value)
        self.global_states.append(global_state)

    def update(self, next_local_state=None, next_global_state=None, done=True):
        """Run ``K_epochs`` of clipped PPO updates over Actor and centralised Critic."""
        if len(self.memory) == 0:
            return
        states, actions, old_log_probs, rewards, dones, values = self.memory.get(DEVICE)

        next_value = 0.0
        if next_global_state is not None and not done:
            gs_tensor = torch.FloatTensor(np.asarray(next_global_state).flatten()).to(DEVICE)
            if gs_tensor.dim() == 1:
                gs_tensor = gs_tensor.unsqueeze(0)
            with torch.no_grad():
                next_value = self.old_critic(gs_tensor).item()

        returns, advantages = self._compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.K_epochs):
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            global_states = torch.FloatTensor(np.array([np.asarray(gs).flatten() for gs in self.global_states])).to(DEVICE)
            state_values = self.critic(global_states).squeeze(-1)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns_normalized)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5
            )
            self.optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        self.memory.clear()
        self.global_states.clear()
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)

    def _compute_gae(self, rewards, values, dones, next_value):
        """GAE processed backwards through the rollout."""
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
        """Persist Actor and Critic weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        """Load Actor and Critic weights. Returns False if the file is missing."""
        if not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        return True


class MAPPO(BaseAgent):
    """MAPPO multi-agent wrapper.

    Each firm has its own Actor and Critic parameters; the global state used by
    each Critic is the concatenation of every firm's local observation.
    """

    def __init__(self, num_agents, state_size, action_size, **kwargs):
        """Create one ``MAPPOAgent`` per controlled firm."""
        super().__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.global_state_size = num_agents * state_size
        self.agents = [
            MAPPOAgent(state_size, action_size, self.global_state_size, **kwargs)
            for _ in range(num_agents)
        ]

    def _global_state(self, local_states):
        """Flatten and concatenate every firm's local observation."""
        return np.concatenate([np.asarray(s).flatten() for s in local_states])

    def act(self, states, explore=True):
        """Have every firm's Actor act in parallel given the shared global state."""
        global_state = self._global_state(states)
        actions = []
        infos = []
        for i, agent in enumerate(self.agents):
            action, a_idx, log_prob, value = agent.act(states[i], global_state, explore=explore)
            actions.append(action)
            infos.append({"action_index": a_idx, "log_prob": log_prob, "value": value})
        return actions, infos

    def step(self, states, infos, rewards, next_states, dones):
        """Push every firm's transition (with global state) into its own buffer."""
        global_state = self._global_state(states)
        next_global_state = self._global_state(next_states)
        for i, agent in enumerate(self.agents):
            agent.step(
                states[i],
                global_state,
                infos[i]["action_index"],
                infos[i]["log_prob"],
                rewards[i],
                next_states[i],
                next_global_state,
                dones,
                value=infos[i]["value"],
            )

    def update(self, next_states=None, done=True):
        """Update every firm in sequence."""
        next_global = self._global_state(next_states) if next_states is not None else None
        for i, agent in enumerate(self.agents):
            ns = next_states[i] if next_states is not None else None
            agent.update(ns, next_global, done)

    def save(self, path_prefix):
        """Persist each firm's Actor + Critic under ``{path_prefix}_agent_{i}.pth``."""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{i}.pth")

    def load(self, path_prefix):
        """Load each firm's Actor + Critic from ``{path_prefix}_agent_{i}.pth``."""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{i}.pth")
        return True