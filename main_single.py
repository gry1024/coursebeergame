"""Single-agent training entry point.

Controls firm ``--firm_id`` (default 1) with a DQN/DDQN/PPO agent; the other
two firms follow the ``order_up_to`` heuristic (or uniform random).
"""

import os
import argparse
import json
import numpy as np
from envs.beer_game_env import BeerGameEnv
from algos.builder import build_agent
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import CSVLogger
from utils.heuristic import order_up_to_policy
from utils import plotting


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-agent supply chain inventory training")
    parser.add_argument("--alg", type=str, default="dqn", choices=["dqn", "ddqn", "ppo"])
    parser.add_argument("--firm_id", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lead_time", type=int, default=0)
    parser.add_argument("--demand", type=str, default="poisson")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heuristic", type=str, default="order_up_to", choices=["order_up_to", "random"])
    return parser.parse_args()


def build_env(config, args):
    """Construct the Beer Game environment from config + CLI overrides."""
    return BeerGameEnv(
        num_firms=config["num_firms"],
        p=config["p"],
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


def choose_heuristic_action(state, firm_id, heuristic_name, max_order):
    """Pick an action for an uncontrolled firm."""
    if heuristic_name == "order_up_to":
        return order_up_to_policy(state[firm_id], firm_id, max_order=max_order)
    return np.random.randint(0, max_order + 1)


def train_dqn_like(agent, env, args, logger):
    """Train a DQN / DDQN agent against heuristic-controlled neighbours."""
    eps = args.eps_start
    scores = []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        score = 0.0
        for _ in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for i in range(env.num_firms):
                if i == args.firm_id:
                    actions[i] = agent.act(state[i], epsilon=eps)
                else:
                    actions[i] = choose_heuristic_action(state, i, args.heuristic, args.max_order)
            next_state, rewards, done = env.step(actions)
            reward = rewards[args.firm_id][0]
            agent.step(state[args.firm_id], actions[args.firm_id], reward, next_state[args.firm_id], done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(args.eps_end, args.eps_decay * eps)
        scores.append(score)
        logger.log({"episode": ep, "score": score, "epsilon": eps})
        if ep % 100 == 0:
            print(f"Episode {ep}/{args.episodes} | Avg Score: {np.mean(scores[-100:]):.2f} | Eps: {eps:.4f}")
        if ep % 500 == 0:
            agent.save(f"models/{args.alg}_firm_{args.firm_id}_ep{ep}.pth")
    return scores


def train_ppo(agent, env, args, logger):
    """Train a PPO agent against heuristic-controlled neighbours."""
    scores = []
    episodes_since_update = 0
    last_state = None
    last_done = True
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        score = 0.0
        for _ in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            action_index = log_prob = value = None
            for i in range(env.num_firms):
                if i == args.firm_id:
                    action, action_index, log_prob, value = agent.act(state[i], explore=True)
                    actions[i] = action
                else:
                    actions[i] = choose_heuristic_action(state, i, args.heuristic, args.max_order)
            next_state, rewards, done = env.step(actions)
            reward = rewards[args.firm_id][0]
            score += reward
            agent.step(state[args.firm_id], action_index, log_prob, reward, next_state[args.firm_id], done, value=value)
            state = next_state
            if done:
                break
        last_state = state[args.firm_id]
        last_done = done
        episodes_since_update += 1
        scores.append(score)
        logger.log({"episode": ep, "score": score})
        if ep % 100 == 0:
            print(f"Episode {ep}/{args.episodes} | Avg Score: {np.mean(scores[-100:]):.2f}")
        if episodes_since_update >= args.update_every:
            agent.update(last_state, last_done)
            episodes_since_update = 0
        if ep % 500 == 0:
            agent.save(f"models/{args.alg}_firm_{args.firm_id}_ep{ep}.pth")
    if episodes_since_update > 0:
        agent.update(last_state, last_done)
    return scores


def test_agent(agent, env, args, num_episodes=10):
    """Run greedy test episodes and return scores, inventories, orders, and bullwhip metrics."""
    scores = []
    inventories = []
    episode_orders = []
    episode_demands = []
    satisfieds = []
    for ep in range(num_episodes):
        state = env.reset()
        score = 0.0
        ep_inv = []
        ep_ord = [[] for _ in range(env.num_firms)]
        ep_dem = [[] for _ in range(env.num_firms)]
        ep_sat = []
        for _ in range(env.max_steps):
            actions = np.zeros((env.num_firms, 1))
            for i in range(env.num_firms):
                if i == args.firm_id:
                    if args.alg in ("dqn", "ddqn"):
                        actions[i] = agent.act(state[i], epsilon=0.0)
                    else:
                        actions[i], _, _, _ = agent.act(state[i], explore=False)
                else:
                    actions[i] = choose_heuristic_action(state, i, args.heuristic, args.max_order)
            next_state, rewards, done = env.step(actions)
            ep_inv.append(env.inventory[args.firm_id][0])
            for i in range(env.num_firms):
                ep_ord[i].append(actions[i][0])
                ep_dem[i].append(env.demand[i][0])
            ep_sat.append(env.satisfied_demand[args.firm_id][0])
            score += rewards[args.firm_id][0]
            state = next_state
            if done:
                break
        scores.append(score)
        inventories.append(ep_inv)
        episode_orders.append(ep_ord)
        episode_demands.append(ep_dem)
        satisfieds.append(ep_sat)
        print(f"Test Episode {ep + 1}/{num_episodes} | Score: {score:.2f}")

    # Per-firm bullwhip ratio: average over test episodes.
    order_vars = []
    demand_vars = []
    bullwhip_ratios = []
    for i in range(env.num_firms):
        ep_order_vars = [np.var(episode_orders[ep][i]) for ep in range(num_episodes)]
        ep_demand_vars = [np.var(episode_demands[ep][i]) for ep in range(num_episodes)]
        order_vars.append(float(np.mean(ep_order_vars)))
        demand_vars.append(float(np.mean(ep_demand_vars)))
        bullwhip_ratios.append(float(np.mean([
            ep_order_vars[ep] / (ep_demand_vars[ep] + 1e-8) for ep in range(num_episodes)
        ])))
    orders_list = [episode_orders[ep][args.firm_id] for ep in range(num_episodes)]
    demands = [episode_demands[ep][args.firm_id] for ep in range(num_episodes)]
    return scores, inventories, orders_list, demands, satisfieds, order_vars, demand_vars, bullwhip_ratios


def main():
    """Parse args, build env + agent, train, test, and save artifacts."""
    args = parse_args()
    set_seed(args.seed)
    plotting.set_style()
    config = load_config(algo_path=f"configs/algo/{args.alg}.yaml")
    # Merge YAML into args so the training functions see a uniform namespace.
    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    env = build_env(config, args)
    state_size = env.obs_dim
    action_size = config["max_order"] + 1
    agent_kwargs = {k: v for k, v in config.items() if k not in ("name", "state_size", "action_size")}
    agent = build_agent(args.alg, state_size, action_size, **agent_kwargs)
    logger = CSVLogger(
        f"results/{args.alg}_single_firm{args.firm_id}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}.csv",
        fieldnames=["episode", "score", "epsilon"] if args.alg in ("dqn", "ddqn") else ["episode", "score"],
    )
    print(f"Training {args.alg.upper()} | firm={args.firm_id} | lead_time={args.lead_time} | demand={args.demand} | seed={args.seed}")
    if args.alg in ("dqn", "ddqn"):
        scores = train_dqn_like(agent, env, args, logger)
    else:
        scores = train_ppo(agent, env, args, logger)
    logger.close()
    agent.save(f"models/{args.alg}_firm_{args.firm_id}_final.pth")
    plotting.plot_training_single(
        scores,
        f"figures/{args.alg}_single_firm{args.firm_id}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}_train.png",
        title=f"{args.alg.upper()} training reward (firm={args.firm_id})",
    )
    test_scores, inv, ord, dem, sat, order_vars, demand_vars, bullwhip_ratios = test_agent(agent, env, args, num_episodes=10)
    plotting.plot_test_dashboard(
        np.mean(inv, axis=0),
        np.mean(ord, axis=0),
        np.mean(dem, axis=0),
        np.mean(sat, axis=0),
        test_scores,
        f"figures/{args.alg}_single_firm{args.firm_id}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}_test.png",
        title=f"{args.alg.upper()} test metrics (firm={args.firm_id})",
    )
    bullwhip_path = f"results/{args.alg}_single_firm{args.firm_id}_lt{args.lead_time}_demand{args.demand}_seed{args.seed}_bullwhip.json"
    with open(bullwhip_path, "w", encoding="utf-8") as f:
        json.dump({"order_vars": order_vars, "demand_vars": demand_vars}, f, ensure_ascii=False, indent=2)
    print("Bullwhip ratio Var(order)/Var(demand):", [f"{bullwhip_ratios[i]:.3f}" for i in range(env.num_firms)])


if __name__ == "__main__":
    main()