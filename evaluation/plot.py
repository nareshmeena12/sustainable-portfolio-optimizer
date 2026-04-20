import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
import sys
sys.path.append(".")

os.makedirs("logs/plots", exist_ok=True)

COLORS = {
    "our_model"   : "#1A1A2E",
    "markowitz"   : "#E94560",
    "vanilla_rl"  : "#0F3460",
    "equal_weight": "#888888",
    "tech"        : "#2ECC71",
    "energy"      : "#F39C12",
    "healthcare"  : "#9B59B6",
}


def plot_training_curves(history_path="logs/train_history.json"):
    with open(history_path) as f:
        history = json.load(f)

    sectors  = list(history[0]["sectors"].keys())
    episodes = [ep["episode"] for ep in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    metrics = ["total_return", "sharpe", "esg_rate"]
    labels  = ["Cumulative Return", "Sharpe Ratio", "ESG Compliance Rate"]

    for ax, metric, label in zip(axes, metrics, labels):
        for sector in sectors:
            vals = [ep["sectors"][sector].get(metric, 0) for ep in history]
            # smooth with rolling average
            smooth = np.convolve(vals, np.ones(20)/20, mode="valid")
            ax.plot(episodes[:len(smooth)], smooth,
                    label=sector, color=COLORS[sector], linewidth=1.5)

        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/plots/training_curves.png", dpi=150)
    plt.close()
    print("Saved: logs/plots/training_curves.png")


def plot_portfolio_values(our_results, baseline_results, sector="tech"):
    """
    Plots cumulative portfolio value over test period for all methods.
    our_results    : dict with pv_history from our agents
    baseline_results: list of dicts from baselines.py
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Portfolio Value — {sector.capitalize()} Sector (2023–2024)",
                 fontsize=13, fontweight="bold")

    # our model
    if sector in our_results:
        pv = our_results[sector]["pv_history"]
        ax.plot(pv, label="Our Model (PINN + Transformer)",
                color=COLORS["our_model"], linewidth=2)

    # baselines
    for r in baseline_results:
        if r["sector"] == sector and "pv_history" in r:
            ax.plot(r["pv_history"], label=r["method"].replace("_", " ").title(),
                    color=COLORS.get(r["method"], "gray"),
                    linewidth=1.5, linestyle="--")

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Initial value")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"logs/plots/portfolio_value_{sector}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sharpe_comparison(our_results, baseline_results):
    sectors = list(our_results.keys())
    methods = ["our_model", "markowitz", "vanilla_rl", "equal_weight"]
    labels  = ["Our Model", "Markowitz", "Vanilla RL", "Equal Weight"]

    x     = np.arange(len(sectors))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Sharpe Ratio Comparison", fontsize=13, fontweight="bold")

    # our model bars
    our_sharpes = [our_results[s]["sharpe"] for s in sectors]
    ax.bar(x - 1.5 * width, our_sharpes, width,
           label="Our Model", color=COLORS["our_model"])

    # baseline bars
    for i, (method, label) in enumerate(zip(methods[1:], labels[1:]), 1):
        sharpes = []
        for sector in sectors:
            match = [r for r in baseline_results
                     if r["method"] == method and r["sector"] == sector]
            sharpes.append(match[0]["sharpe"] if match else 0)
        ax.bar(x + (i - 1.5) * width, sharpes, width,
               label=label, color=COLORS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sectors])
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("logs/plots/sharpe_comparison.png", dpi=150)
    plt.close()
    print("Saved: logs/plots/sharpe_comparison.png")


def plot_esg_compliance(our_results, baseline_results):
    sectors = list(our_results.keys())
    methods = ["our_model", "markowitz", "vanilla_rl", "equal_weight"]
    labels  = ["Our Model", "Markowitz", "Vanilla RL", "Equal Weight"]

    x     = np.arange(len(sectors))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("ESG Compliance Rate (% days above threshold 0.6)",
                 fontsize=13, fontweight="bold")

    our_esg = [our_results[s]["esg_rate"] * 100 for s in sectors]
    ax.bar(x - 1.5 * width, our_esg, width,
           label="Our Model", color=COLORS["our_model"])

    for i, (method, label) in enumerate(zip(methods[1:], labels[1:]), 1):
        esgs = []
        for sector in sectors:
            match = [r for r in baseline_results
                     if r["method"] == method and r["sector"] == sector]
            esgs.append(match[0]["esg_rate"] * 100 if match else 0)
        ax.bar(x + (i - 1.5) * width, esgs, width,
               label=label, color=COLORS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sectors])
    ax.set_ylabel("ESG Compliance (%)")
    ax.set_ylim(0, 110)
    ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="Target 80%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/plots/esg_compliance.png", dpi=150)
    plt.close()
    print("Saved: logs/plots/esg_compliance.png")


def plot_ablation(ablation_results):
    """
    ablation_results: dict like
    {
      "full_model"         : {"sharpe": x, "esg_rate": y},
      "no_pinn"            : {"sharpe": x, "esg_rate": y},
      "no_transformer"     : {"sharpe": x, "esg_rate": y},
      "no_esg"             : {"sharpe": x, "esg_rate": y},
    }
    """
    labels  = list(ablation_results.keys())
    sharpes = [ablation_results[k]["sharpe"]   for k in labels]
    esgs    = [ablation_results[k]["esg_rate"] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation Study", fontsize=13, fontweight="bold")

    bar_colors = [COLORS["our_model"]] + [COLORS["vanilla_rl"]] * (len(labels) - 1)

    ax1.bar(labels, sharpes, color=bar_colors)
    ax1.set_title("Sharpe Ratio")
    ax1.set_ylabel("Sharpe")
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=15)

    ax2.bar(labels, [e * 100 for e in esgs], color=bar_colors)
    ax2.set_title("ESG Compliance Rate")
    ax2.set_ylabel("ESG %")
    ax2.axhline(80, color="red", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("logs/plots/ablation.png", dpi=150)
    plt.close()
    print("Saved: logs/plots/ablation.png")


if __name__ == "__main__":
    # test with dummy data so you can verify plots render correctly
    dummy_history = [
        {"episode": i, "sectors": {
            "tech":       {"total_return": np.random.randn()*0.01, "sharpe": np.random.randn()*0.5, "esg_rate": np.random.uniform(0.5, 0.9)},
            "energy":     {"total_return": np.random.randn()*0.01, "sharpe": np.random.randn()*0.5, "esg_rate": np.random.uniform(0.4, 0.8)},
            "healthcare": {"total_return": np.random.randn()*0.01, "sharpe": np.random.randn()*0.5, "esg_rate": np.random.uniform(0.6, 0.95)},
        }} for i in range(1, 101)
    ]

    with open("logs/train_history.json", "w") as f:
        json.dump(dummy_history, f)

    plot_training_curves()

    # dummy ablation
    plot_ablation({
        "Full Model"     : {"sharpe": 0.82, "esg_rate": 0.84},
        "No PINN"        : {"sharpe": 0.61, "esg_rate": 0.80},
        "No Transformer" : {"sharpe": 0.70, "esg_rate": 0.81},
        "No ESG"         : {"sharpe": 0.78, "esg_rate": 0.42},
    })

    print("plot.py OK")