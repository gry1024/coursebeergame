"""Beer Game supply chain environment.

A configurable multi-echelon inventory simulator. Each step firms submit
integer order quantities, the environment generates downstream demand,
satisfies it against current inventory, and updates state. Supports an
optional FIFO pipeline for non-zero lead time.
"""

import numpy as np
from collections import deque


class BeerGameEnv:
    """Serial supply chain with ``num_firms`` stages."""

    def __init__(
        self,
        num_firms,
        p,
        h,
        c,
        initial_inventory,
        poisson_lambda=10,
        max_steps=100,
        lead_time=0,
        demand_type="poisson",
        seasonal_amplitude=5.0,
        seasonal_period=25,
        shock_prob=0.05,
        shock_multiplier=4.0,
        max_order=20,
        history_len=None,
    ):
        """Initialise parameters and per-step state.

        ``lead_time`` orders arriving ``L`` steps later are kept in per-firm
        FIFO queues (``pipeline``). When ``lead_time == 0`` orders arrive
        immediately, so ``pipeline`` is unused.
        """
        self.num_firms = num_firms
        self.p = np.asarray(p).reshape(-1, 1)
        self.h = h
        self.c = c
        self.poisson_lambda = poisson_lambda
        self.max_steps = max_steps
        self.lead_time = lead_time
        self.demand_type = demand_type
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.shock_prob = shock_prob
        self.shock_multiplier = shock_multiplier
        self.initial_inventory = initial_inventory

        # In-transit FIFO queue per firm; unused when lead_time == 0.
        self.pipeline = [
            deque([0.0] * lead_time, maxlen=lead_time) for _ in range(num_firms)
        ] if lead_time > 0 else None

        self.inventory = np.full((num_firms, 1), initial_inventory, dtype=float)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((num_firms, 1))
        self.demand = np.zeros((num_firms, 1))
        self.current_step = 0
        self.done = False

    @property
    def obs_dim(self):
        """Per-firm observation dimension: 3 without lead time, 4 with."""
        return 4 if self.lead_time > 0 else 3

    def reset(self):
        """Reset all per-step state and the in-transit queues."""
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory, dtype=float)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((num_firms := self.num_firms, 1)) if False else np.zeros((self.num_firms, 1))
        self.demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        if self.lead_time > 0:
            self.pipeline = [
                deque([0.0] * self.lead_time, maxlen=self.lead_time)
                for _ in range(self.num_firms)
            ]
        return self._get_observation()

    def _get_observation(self):
        """Build per-firm observation: [order, satisfied_demand, inventory, (pipeline_inventory if L>0)]."""
        obs = np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)
        if self.lead_time > 0:
            pipeline = np.array([[sum(q) for q in self.pipeline]]).T
            obs = np.concatenate((obs, pipeline), axis=1)
        return obs

    def _current_lambda(self):
        """Demand intensity for the current step based on ``demand_type``."""
        if self.demand_type == "poisson":
            return self.poisson_lambda
        if self.demand_type == "seasonal":
            lam = self.poisson_lambda + self.seasonal_amplitude * np.sin(
                2 * np.pi * self.current_step / self.seasonal_period
            )
            # Floor at a small positive value to keep Poisson sampling valid.
            return max(lam, 0.1)
        if self.demand_type == "shock":
            return self.poisson_lambda
        raise ValueError(f"Unknown demand_type: {self.demand_type}")

    def _generate_demand(self):
        """Sample firm 0's external demand; pass each downstream order to the upstream firm."""
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                lam = self._current_lambda()
                d = np.random.poisson(lam)
                # Optional shock: with probability ``shock_prob`` add a large jump.
                if self.demand_type == "shock" and np.random.random() < self.shock_prob:
                    d = int(lam + self.shock_multiplier * self.poisson_lambda)
                demand[i] = d
            else:
                demand[i] = self.orders[i - 1]
        return demand

    def _arrivals(self):
        """Compute the arrival quantity this step: orders placed ``L`` steps ago, or current orders when L==0."""
        if self.lead_time == 0:
            return self.orders.copy()
        arrivals = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            arrivals[i] = self.pipeline[i].popleft()
        return arrivals

    def _push_pipeline(self):
        """Append the current step's orders to the back of each pipeline."""
        if self.lead_time > 0:
            for i in range(self.num_firms):
                self.pipeline[i].append(float(self.orders[i]))

    def step(self, actions):
        """Advance one timestep and return (next_obs, rewards, done)."""
        # Record the orders submitted by each firm this step.
        self.orders = np.asarray(actions).reshape(self.num_firms, 1)
        self.demand = self._generate_demand()
        arrivals = self._arrivals()
        # Satisfy as much demand as possible from current inventory.
        self.satisfied_demand = np.minimum(self.demand, self.inventory)
        self._push_pipeline()
        # Inventory update: + arrivals - satisfied demand.
        self.inventory = self.inventory + arrivals - self.satisfied_demand
        # Per-firm reward: revenue - upstream cost - holding - lost sales.
        rewards = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            revenue = self.p[i] * self.satisfied_demand[i]
            purchase = (self.p[i + 1] if i + 1 < self.num_firms else 0.0) * self.orders[i]
            holding = self.h * self.inventory[i]
            rewards[i] = revenue - purchase - holding
        loss_sales = np.where(
            self.satisfied_demand < self.demand,
            (self.demand - self.satisfied_demand) * self.c,
            0.0,
        )
        rewards -= loss_sales

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        return self._get_observation(), rewards, self.done