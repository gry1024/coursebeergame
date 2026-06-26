"""Factory function for constructing agents by name."""

from algos.dqn.agent import DQNAgent
from algos.ddqn.agent import DDQNAgent
from algos.ppo_ippo.agent import PPOAgent, IPPOAgent
from algos.mappo.agent import MAPPO


def build_agent(algo_name, state_size, action_size, **kwargs):
    """Instantiate an agent from its string identifier.

    Multi-agent variants (``ippo``, ``mappo``) additionally require
    ``num_agents`` in ``kwargs``, which is popped before passing the rest
    down to the underlying constructors.
    """
    algo_name = algo_name.lower()
    if algo_name == "dqn":
        return DQNAgent(state_size, action_size, **kwargs)
    if algo_name == "ddqn":
        return DDQNAgent(state_size, action_size, **kwargs)
    if algo_name == "ppo":
        return PPOAgent(state_size, action_size, **kwargs)
    if algo_name == "ippo":
        num_agents = kwargs.pop("num_agents")
        return IPPOAgent(num_agents, state_size, action_size, **kwargs)
    if algo_name == "mappo":
        num_agents = kwargs.pop("num_agents")
        return MAPPO(num_agents, state_size, action_size, **kwargs)
    raise ValueError(f"Unknown algorithm: {algo_name}")