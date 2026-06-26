"""Multi-agent training entry point.

Trains IPPO or MAPPO with all ``--num_agents`` firms controlled by independent
(or centralised-critic) PPO agents.
"""

import argparse
import json
import numpy as np
from envs.beer_game_env import BeerGameEnv
from algos.builder import build_agent
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import CSVLogger
from utils import plotting


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-agent supply chain inventory training")
    parser.add_argument("--alg", type=str, default="ippo", choices=["ippo", "mappo"])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--lead_time", type=int, default=0)
    parser.add_argument("--demand", type=str, default="poisson")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_env(config, args):
    """Construct the environment; ``num_agents`` equals ``num_firms``."""
    return BeerGameEnv(
        num_firms=args.num_agents,
        p=config["p"][:args.num_agents],
        h=config["h"],
        c=config["c"],
        initial_inventory=config["initial_inventory"],
        poisson_lambda=config["poisson_lambda"],
        max_steps=config["max_steps"],
        lead_time=args.lead_time,
        demand_type=args.demand,
        max_order=config["max_order"],
        history_len=config.get("history_len"),
    )


def train_multi(agent, env, args, logger):
    """Train IPPO / MAPPO. The team reward (sum across firms) is used as the per-agent signal."""
    scores = []
    episodes_since_update = 0
    last_states = None
    last_done = True
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        score = 0.0
        for _ in range(env.max_steps):
            actions, infos = agent.act(state, explore=True)
            actions_arr = np.array(actions).reshape(env.num_firms, 1)
            next_state, rewards, done = env.step(actions_arr)
            team_reward = rewards.sum()
            score += team_reward
            agent.step(state, infos, rewards.flatten(), next_state, done)
            state = next_state
            if done:
                break
        last_states = state
        last_done = done
        episodes_since_update += 1
        scores.append(score)
        logger.log({"episode": ep, "score": score})
        if ep % 100 == 0:
            print(f"Episode {ep}/{args.episodes} | Avg Score: {np.mean(scores[-100:]):.2f}")
        if episodes_since_update >= args.update_every:
            agent.update(last_states, last_done)
            episodes_since_update = 0
        if ep % 500 == 0:
            agent.save(f"models/{args.alg}_ep{ep}")
    if episodes_since_update > 0:
        agent.update(last_states, last_done)
    return scores


def test_multi(agent, env, args, num_episodes=10):
    """Run greedy test episodes and record per-firm orders and demands."""
    scores = []
    episode_orders = []
    episode_demands = []
    for ep in range(num_episodes):
        state = env.reset()
        score = 0.0
        ep_orders = [[] for _ in range(env.num_firms)]
        ep_demands = [[] for _ in range(env.num_firms)]
        for _ in range(env.max_steps):
            actions, _ = agent.act(state, explore=False)
            actions_arr = np.array(actions).reshape(env.num_firms, 1)
            next_state, rewards, done = env.step(actions_arr)
            score += rewards.sum()
            for i in range(env.num_firms):
                ep_orders[i].append(actions_arr[i][0])
                ep_demands[i].append(env.demand[i][0])
            state = next_state
            if done:
                break
        scores.append(score)
        episode_orders.append(ep_orders)
        episode_demands.append(ep_demands)
        print(f"Test Episode {ep + 1}/{num_episodes} | Score: {score:.2f}")
    return scores, episode_orders, episode_demands


def main():
    """Parse args, build env + agent, train, test, and save artifacts."""
    args = parse_args()
    set_seed(args.seed)
    plotting.set_style()
    config = load_config(algo_path=f"configs/algo/{args.alg}.yaml")
    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    env = build_env(config, args)
    state_size = env.obs_dim
    action_size = config["max_order"] + 1
    agent_kwargs = {k: v for k, v in config.items() if k not in ("name", "state_size", "action_size")}
    agent_kwargs["num_agents"] = args.num_agents
    agent = build_agent(args.alg, state_size, action_size, **agent_kwargs)
    logger = CSVLogger(
        f"results/{args.alg}_n{args.num_agents}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}.csv",
        fieldnames=["episode", "score"],
    )
    print(f"Training {args.alg.upper()} | num_agents={args.num_agents} | lead_time={args.lead_time} | demand={args.demand} | seed={args.seed}")
    scores = train_multi(agent, env, args, logger)
    logger.close()
    agent.save(f"models/{args.alg}_n{args.num_agents}_final")
    plotting.plot_training_single(
        scores,
        f"figures/{args.alg}_n{args.num_agents}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}_train.png",
        title=f"{args.alg.upper()} training reward (n={args.num_agents})",
    )
    test_scores, episode_orders, episode_demands = test_multi(agent, env, args, num_episodes=10)

    # Per-firm bullwhip ratio averaged over test episodes.
    num_eps = len(episode_orders)
    order_vars = []
    demand_vars = []
    bullwhip_ratios = []
    for i in range(env.num_firms):
        ep_order_vars = [np.var(episode_orders[ep][i]) for ep in range(num_eps)]
        ep_demand_vars = [np.var(episode_demands[ep][i]) for ep in range(num_eps)]
        order_vars.append(float(np.mean(ep_order_vars)))
        demand_vars.append(float(np.mean(ep_demand_vars)))
        bullwhip_ratios.append(float(np.mean([
            ep_order_vars[ep] / (ep_demand_vars[ep] + 1e-8) for ep in range(num_eps)
        ])))
    bullwhip_path = f"results/{args.alg}_n{args.num_agents}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}_bullwhip.json"
    with open(bullwhip_path, "w", encoding="utf-8") as f:
        json.dump({"order_vars": order_vars, "demand_vars": demand_vars}, f, ensure_ascii=False, indent=2)
    print("Bullwhip ratio Var(order)/Var(demand):", [f"{bullwhip_ratios[i]:.3f}" for i in range(env.num_firms)])


if __name__ == "__main__":
    main()