"""Batched grid-experiment driver.

For every combination of (algorithm x lead_time x demand x seed) the driver
either runs the corresponding ``main_single`` / ``main_multi`` script or
re-uses an existing CSV. After training it aggregates the CSVs into three
kinds of figures:

* learning curves with shaded seed variance,
* environment-sensitivity heatmaps,
* bullwhip-effect bar charts.
"""

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
from utils import plotting


# Single-agent and multi-agent algorithm families.
SINGLE_ALGS = {"dqn", "ddqn", "ppo"}
MULTI_ALGS = {"ippo", "mappo"}


def parse_args():
    """Parse the comparison-script command line."""
    parser = argparse.ArgumentParser(description="Batched grid experiment driver")
    parser.add_argument("--algs", type=str, default="dqn,ddqn,ppo")
    parser.add_argument("--firm_id", type=int, default=1)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--lead_times", type=str, default="0")
    parser.add_argument("--demands", type=str, default="poisson")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--skip_run", action="store_true", help="Only plot from existing CSV results")
    return parser.parse_args()


def run_single_experiment(alg, firm_id, lead_time, demand, seed, episodes):
    """Invoke ``main_single.py`` for one configuration."""
    cmd = [
        sys.executable, "main_single.py",
        "--alg", alg,
        "--firm_id", str(firm_id),
        "--lead_time", str(lead_time),
        "--demand", demand,
        "--seed", str(seed),
        "--episodes", str(episodes),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_multi_experiment(alg, num_agents, lead_time, demand, seed, episodes):
    """Invoke ``main_multi.py`` for one configuration."""
    cmd = [
        sys.executable, "main_multi.py",
        "--alg", alg,
        "--num_agents", str(num_agents),
        "--lead_time", str(lead_time),
        "--demand", demand,
        "--seed", str(seed),
        "--episodes", str(episodes),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def single_result_path(alg, firm_id, lead_time, demand, seed):
    """Path to a single-agent training CSV."""
    return f"results/{alg}_single_firm{firm_id}_lt{lead_time}_demand{demand}_seed{seed}.csv"


def multi_result_path(alg, num_agents, lead_time, demand, seed):
    """Path to a multi-agent training CSV."""
    return f"results/{alg}_n{num_agents}_lt{lead_time}_demand{demand}_seed{seed}.csv"


def load_scores(path):
    """Read the ``score`` column from a training CSV. Returns ``None`` if missing."""
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df["score"].values


def single_bullwhip_path(alg, firm_id, lead_time, demand, seed):
    """Path to a single-agent bullwhip JSON."""
    return f"results/{alg}_single_firm{firm_id}_lt{lead_time}_demand{demand}_seed{seed}_bullwhip.json"


def multi_bullwhip_path(alg, num_agents, lead_time, demand, seed):
    """Path to a multi-agent bullwhip JSON."""
    return f"results/{alg}_n{num_agents}_lt{lead_time}_demand{demand}_seed{seed}_bullwhip.json"


def load_bullwhip(path):
    """Read a bullwhip JSON and return ``(order_vars, demand_vars)`` or ``(None, None)``."""
    if not os.path.isfile(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["order_vars"], data["demand_vars"]


def main():
    """Run the requested grid and write the aggregated figures."""
    args = parse_args()
    plotting.set_style()
    algs = [a.strip().lower() for a in args.algs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    lead_times = [int(l) for l in args.lead_times.split(",")]
    demands = [d.strip() for d in args.demands.split(",")]

    # 1. Run experiments unless --skip_run was passed.
    if not args.skip_run:
        for alg in algs:
            for lt in lead_times:
                for demand in demands:
                    for seed in seeds:
                        if alg in SINGLE_ALGS:
                            run_single_experiment(alg, args.firm_id, lt, demand, seed, args.episodes)
                        elif alg in MULTI_ALGS:
                            run_multi_experiment(alg, args.num_agents, lt, demand, seed, args.episodes)
                        else:
                            raise ValueError(f"Unknown algorithm: {alg}")

    # 2. Plot learning curves per (lead_time, demand), split by single/multi.
    single_algs = [a for a in algs if a in SINGLE_ALGS]
    multi_algs = [a for a in algs if a in MULTI_ALGS]

    for lt in lead_times:
        for demand in demands:
            # Single-agent learning curves.
            single_data = {}
            for alg in single_algs:
                seeds_scores = []
                for seed in seeds:
                    path = single_result_path(alg, args.firm_id, lt, demand, seed)
                    scores = load_scores(path)
                    if scores is not None:
                        seeds_scores.append(scores)
                if seeds_scores:
                    single_data[alg] = seeds_scores
            if single_data:
                plotting.plot_learning_curves(
                    single_data,
                    f"figures/compare_single_lt{lt}_demand{demand}_firm{args.firm_id}_learning.png",
                    title=f"Single-agent learning curves (lead_time={lt}, demand={demand})",
                )
            # Multi-agent learning curves.
            multi_data = {}
            for alg in multi_algs:
                seeds_scores = []
                for seed in seeds:
                    path = multi_result_path(alg, args.num_agents, lt, demand, seed)
                    scores = load_scores(path)
                    if scores is not None:
                        seeds_scores.append(scores)
                if seeds_scores:
                    multi_data[alg] = seeds_scores
            if multi_data:
                plotting.plot_learning_curves(
                    multi_data,
                    f"figures/compare_multi_lt{lt}_demand{demand}_n{args.num_agents}_learning.png",
                    title=f"Multi-agent learning curves (lead_time={lt}, demand={demand})",
                )

    # 3. Sensitivity heatmaps: one per algorithm, demand x lead_time.
    if len(demands) > 1 or len(lead_times) > 1:
        for alg in single_algs:
            rows = []
            for demand in demands:
                row = []
                for lt in lead_times:
                    vals = []
                    for seed in seeds:
                        path = single_result_path(alg, args.firm_id, lt, demand, seed)
                        scores = load_scores(path)
                        if scores is not None:
                            vals.append(np.mean(scores[-100:]))
                    row.append(np.mean(vals) if vals else 0.0)
                rows.append(row)
            df = pd.DataFrame(rows, index=demands, columns=lead_times)
            plotting.plot_sensitivity_heatmap(
                df,
                f"figures/compare_{alg}_firm{args.firm_id}_sensitivity.png",
                title=f"{alg.upper()} environment sensitivity",
            )
        for alg in multi_algs:
            rows = []
            for demand in demands:
                row = []
                for lt in lead_times:
                    vals = []
                    for seed in seeds:
                        path = multi_result_path(alg, args.num_agents, lt, demand, seed)
                        scores = load_scores(path)
                        if scores is not None:
                            vals.append(np.mean(scores[-100:]))
                    row.append(np.mean(vals) if vals else 0.0)
                rows.append(row)
            df = pd.DataFrame(rows, index=demands, columns=lead_times)
            plotting.plot_sensitivity_heatmap(
                df,
                f"figures/compare_{alg}_n{args.num_agents}_sensitivity.png",
                title=f"{alg.upper()} environment sensitivity",
            )

    # 4. Bullwhip-effect bar charts.
    for lt in lead_times:
        for demand in demands:
            # Single agent.
            single_order_vars = {}
            single_demand_vars = []
            for alg in single_algs:
                alg_order_vars = []
                alg_demand_vars = []
                for seed in seeds:
                    path = single_bullwhip_path(alg, args.firm_id, lt, demand, seed)
                    ovars, dvars = load_bullwhip(path)
                    if ovars is not None:
                        alg_order_vars.append(ovars)
                        alg_demand_vars.append(dvars)
                if alg_order_vars:
                    single_order_vars[alg] = np.mean(alg_order_vars, axis=0)
                    single_demand_vars.extend(alg_demand_vars)
            if single_order_vars:
                demand_var_mean = np.mean(single_demand_vars, axis=0)
                plotting.plot_bullwhip(
                    single_order_vars,
                    demand_var_mean,
                    f"figures/compare_bullwhip_single_lt{lt}_demand{demand}_firm{args.firm_id}.png",
                    title=f"Single-agent bullwhip effect (lead_time={lt}, demand={demand})",
                )
            # Multi agent.
            multi_order_vars = {}
            multi_demand_vars = []
            for alg in multi_algs:
                alg_order_vars = []
                alg_demand_vars = []
                for seed in seeds:
                    path = multi_bullwhip_path(alg, args.num_agents, lt, demand, seed)
                    ovars, dvars = load_bullwhip(path)
                    if ovars is not None:
                        alg_order_vars.append(ovars)
                        alg_demand_vars.append(dvars)
                if alg_order_vars:
                    multi_order_vars[alg] = np.mean(alg_order_vars, axis=0)
                    multi_demand_vars.extend(alg_demand_vars)
            if multi_order_vars:
                demand_var_mean = np.mean(multi_demand_vars, axis=0)
                plotting.plot_bullwhip(
                    multi_order_vars,
                    demand_var_mean,
                    f"figures/compare_bullwhip_multi_lt{lt}_demand{demand}_n{args.num_agents}.png",
                    title=f"Multi-agent bullwhip effect (lead_time={lt}, demand={demand})",
                )


if __name__ == "__main__":
    main()