"""Abstract base class for all RL agents in this project."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Common interface shared by every agent implementation.

    Subclasses must implement ``act``, ``step``, ``update``, ``save`` and ``load``.
    Concrete semantics (e.g. what ``step`` stores, when ``update`` is called)
    are agent-specific; see each implementation.
    """

    @abstractmethod
    def act(self, state, **kwargs):
        """Select an action given an observation."""
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        """Buffer the current transition (and, for DQN/DDQN, optionally learn)."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Run a policy / value network update from buffered data."""
        pass

    @abstractmethod
    def save(self, path):
        """Persist network weights to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Restore network weights from disk."""
        pass