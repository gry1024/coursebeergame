"""Plotting utilities for training curves, test dashboards, bullwhip, and sensitivity."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False
import seaborn as sns


# Fixed colour per algorithm so figures are visually consistent across reports.
ALGO_COLORS = {
    "dqn": "#7f7f7f",
    "ddqn": "#1f77b4",
    "ppo": "#ff7f0e",
    "ippo": "#2ca02c",
    "mappo": "#d62728",
}


def _available_fonts():
    """Return the set of font names installed on the system."""
    return {f.name for f in fm.fontManager.ttflist}


def set_style():
    """Apply a unified matplotlib style with CJK font fallback.

    Uses SciencePlots' ``science`` + ``no-latex`` style when available so that
    CJK glyphs render without invoking the LaTeX engine. Falls back to DejaVu
    Serif if no CJK font is found.
    """
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "no-latex"])

    available = _available_fonts()
    cjk_candidates = ["Noto Serif SC", "Noto Sans SC", "Source Han Serif SC", "SimSun", "SimHei"]
    cjk_font = next((f for f in cjk_candidates if f in available), None)

    serif_fallback = ["DejaVu Serif", "Times New Roman"]
    if cjk_font:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [cjk_font] + serif_fallback
    else:
        plt.rcParams["font.family"] = ["DejaVu Serif"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def _save(path):
    """Save the current figure and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def moving_average(data, window=100):
    """Compute a centred moving average with the same length as the input.

    Positions before the first full window are filled with a running mean so
    the returned array has the same length as ``data``.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return data
    if n < window:
        return np.cumsum(data) / np.arange(1, n + 1)
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ma = ret[window - 1:] / window
    prefix = np.cumsum(data[:window - 1]) / np.arange(1, window)
    return np.concatenate([prefix, ma])


def plot_learning_curves(data_dict, save_path, window=100, title="学习曲线"):
    """Plot learning curves for multiple algorithms and seeds.

    Each algorithm's seeds are aligned to the shortest run length, smoothed
    individually with a moving average, then summarised by mean and standard
    deviation (rendered as a shaded band).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, seeds in data_dict.items():
        min_len = min(len(s) for s in seeds)
        arr = np.vstack([s[:min_len] for s in seeds])
        smoothed = np.vstack([moving_average(arr[i], window) for i in range(arr.shape[0])])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean))
        color = ALGO_COLORS.get(algo, None)
        ax.plot(x, mean, label=algo.upper(), color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="best", frameon=True)
    _save(save_path)


def plot_test_dashboard(inventory, orders, demand, satisfied, rewards, save_path, title="测试指标"):
    """Plot a four-panel test summary: inventory, orders, demand vs satisfied, reward boxplot."""
    set_style()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(inventory, linewidth=1.5)
    axs[0, 0].set_title("库存水平")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Inventory")
    axs[0, 1].plot(orders, linewidth=1.5)
    axs[0, 1].set_title("订单量")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Order")
    axs[1, 0].plot(demand, label="Demand", linewidth=1.5)
    axs[1, 0].plot(satisfied, label="Satisfied", linewidth=1.5)
    axs[1, 0].set_title("需求 vs 满足需求")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Quantity")
    axs[1, 0].legend(frameon=True)
    axs[1, 1].boxplot(rewards)
    axs[1, 1].set_title("测试奖励分布")
    axs[1, 1].set_ylabel("Total Reward")
    fig.suptitle(title)
    plt.tight_layout()
    _save(save_path)


def plot_order_behavior(order_dict, demand, save_path, title="最终策略订购行为对比"):
    """Overlay each algorithm's average order curve over the demand curve."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(demand, label="Demand", color="black", linestyle="--", alpha=0.6, linewidth=1.5)
    for algo, orders in order_dict.items():
        color = ALGO_COLORS.get(algo, None)
        ax.plot(orders, label=algo.upper(), color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Order Quantity")
    ax.legend(loc="best", frameon=True)
    _save(save_path)


def plot_bullwhip(order_var_dict, demand_var, save_path, title="牛鞭效应对比"):
    """Grouped bar chart of Var(Order) / Var(Demand) per firm per algorithm."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(demand_var))
    n_algos = len(order_var_dict)
    width = 0.8 / max(n_algos, 1)
    for idx, (algo, var) in enumerate(order_var_dict.items()):
        color = ALGO_COLORS.get(algo, None)
        ratio = np.array(var) / (demand_var + 1e-8)
        ax.bar(x + idx * width, ratio, width, label=algo.upper(), color=color)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Firm ID")
    ax.set_ylabel("Var(Order) / Var(Demand)")
    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([f"Firm {i}" for i in x])
    ax.legend(loc="best", frameon=True)
    _save(save_path)


def plot_sensitivity_heatmap(data, save_path, title="环境敏感性热力图"):
    """Heatmap of average test reward over (lead_time x demand_type)."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Avg Reward"})
    ax.set_title(title)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Demand Type")
    _save(save_path)


def plot_training_single(scores, save_path, window=100, title="训练奖励"):
    """Plot a single training run: raw reward (faded) and moving average."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = np.asarray(scores)
    avg = moving_average(scores, window)
    ax.plot(np.arange(len(scores)), scores, alpha=0.3, label="Raw", linewidth=0.8)
    ax.plot(np.arange(len(avg)), avg, label=f"MA{window}", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(frameon=True)
    _save(save_path)