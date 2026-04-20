import os
import json
import pickle
import torch
import numpy as np
import sys
sys.path.append(".")

from marl.agent_manager import AgentManager

# ── config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # hardware
    "device"      : torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # training
    "n_episodes"  : 500,
    "max_steps"   : 252,    # one trading year per episode

    # ppo
    "n_epochs"    : 4,
    "batch_size"  : 64,
    "lr_actor"    : 3e-4,
    "lr_critic"   : 1e-3,
    "gamma"       : 0.99,
    "lam"         : 0.95,
    "clip_eps"    : 0.2,
    "entropy_c"   : 0.01,

    # environment
    "esg_lambda"      : 0.3,
    "esg_drift_std"   : 0.002,
    "transaction_cost": 0.001,
    "max_drawdown"    : 0.5,

    # pinn
    "pinn_alpha"  : 0.1,
    "hidden_dim"  : 128,
    "n_layers"    : 4,
    "pinn_path"   : "checkpoints/pinn_pretrained.pt",
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)


def load_data():
    with open("data/processed/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("data/processed/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"Data loaded | sectors: {list(train_data.keys())}")
    return train_data, test_data


def save_history(history, path="logs/train_history.json"):
    # convert numpy types to python native for json serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    with open(path, "w") as f:
        json.dump(history, f, default=convert, indent=2)
    print(f"Training history saved to {path}")


def print_final_summary(history, sectors):
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — FINAL 10 EPISODE AVERAGE")
    print("=" * 60)

    last_10 = history[-10:]

    for sector in sectors:
        returns = [ep["sectors"][sector].get("total_return", 0) for ep in last_10]
        sharpes = [ep["sectors"][sector].get("sharpe", 0)       for ep in last_10]
        esgs    = [ep["sectors"][sector].get("esg_rate", 0)     for ep in last_10]

        print(f"\n  {sector.upper()}")
        print(f"    Avg Return : {np.mean(returns):.4f}")
        print(f"    Avg Sharpe : {np.mean(sharpes):.4f}")
        print(f"    ESG Rate   : {np.mean(esgs):.2%}")

    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  ESG-PINN PORTFOLIO — TRAINING")
    print(f"  Device  : {CONFIG['device']}")
    print(f"  Episodes: {CONFIG['n_episodes']}")
    print("=" * 60)

    train_data, test_data = load_data()

    manager = AgentManager(train_data, CONFIG)

    # run full training
    history = manager.train(CONFIG["n_episodes"])

    # save final model and history
    manager.save("checkpoints/final")
    save_history(history)

    print_final_summary(history, manager.sectors)

    print("\nNext: run evaluation/metrics.py")


if __name__ == "__main__":
    main()