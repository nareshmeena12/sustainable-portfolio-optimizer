import numpy as np
import pickle
import torch
import sys
sys.path.append(".")

from env.portfolio_env import PortfolioEnv
from evaluation.metrics import sharpe_ratio, max_drawdown, cumulative_return, esg_compliance_rate


# ── Markowitz ─────────────────────────────────────────────────────────────────

def markowitz_weights(train_returns, risk_aversion=2.0):
    """
    Mean-variance optimal weights from training data.
    Computed once and held fixed during test — classic Markowitz.
    """
    mu    = train_returns.mean(axis=0)
    sigma = np.cov(train_returns.T) + np.eye(train_returns.shape[1]) * 1e-6

    # analytical solution: w = (1/lambda) * Sigma^-1 * mu, normalized
    w = np.linalg.solve(risk_aversion * sigma, mu)
    w = np.clip(w, 0, None)           # no shorting
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    return w


def run_markowitz(test_data, train_data, sector):
    # compute weights from training returns
    train_y = train_data[sector]["y"]    # (T, n_stocks)
    weights = markowitz_weights(train_y)

    # add cash = 0 to match action space
    weights_full = np.append(weights * 0.95, 0.05)  # 5% cash

    env    = PortfolioEnv(test_data, sector)
    obs, _ = env.reset()

    pv_history     = [1.0]
    ret_history    = []
    weight_history = []
    done           = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(weights_full)
        done = terminated or truncated
        ret_history.append(info["daily_return"])
        pv_history.append(info["portfolio_value"])
        weight_history.append(info["weights"])

    esg_base = test_data[sector]["esg"]

    return {
        "method"       : "markowitz",
        "sector"       : sector,
        "cum_return"   : round(cumulative_return(pv_history),                       4),
        "sharpe"       : round(sharpe_ratio(ret_history),                           4),
        "max_drawdown" : round(max_drawdown(pv_history),                            4),
        "esg_rate"     : round(esg_compliance_rate(weight_history, esg_base),       4),
        "pv_history"   : pv_history,
    }


# ── Vanilla RL (MLP actor, no PINN, no ESG) ───────────────────────────────────

class MLPActor(torch.nn.Module):
    """Plain MLP actor — same action space as Transformer actor but no attention."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def run_vanilla_rl(test_data, sector, obs_dim, n_stocks, device="cpu"):
    """
    Runs a randomly initialized MLP actor (no training) as the vanilla RL baseline.
    In a full ablation you'd train this properly — for quick comparison
    a random MLP shows the floor performance without physics or ESG.
    
    To get proper vanilla RL results: train the same PPO setup
    but replace TransformerActor with MLPActor and remove ESG from reward.
    """
    actor  = MLPActor(obs_dim, n_stocks + 1).to(device)
    env    = PortfolioEnv(test_data, sector)
    obs, _ = env.reset()

    pv_history     = [1.0]
    ret_history    = []
    weight_history = []
    done           = False

    while not done:
        with torch.no_grad():
            t_obs   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            weights = actor(t_obs).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(weights)
        done = terminated or truncated
        ret_history.append(info["daily_return"])
        pv_history.append(info["portfolio_value"])
        weight_history.append(info["weights"])

    esg_base = test_data[sector]["esg"]

    return {
        "method"       : "vanilla_rl",
        "sector"       : sector,
        "cum_return"   : round(cumulative_return(pv_history),                 4),
        "sharpe"       : round(sharpe_ratio(ret_history),                     4),
        "max_drawdown" : round(max_drawdown(pv_history),                      4),
        "esg_rate"     : round(esg_compliance_rate(weight_history, esg_base), 4),
        "pv_history"   : pv_history,
    }


# ── Equal weight ──────────────────────────────────────────────────────────────

def run_equal_weight(test_data, sector):
    """Simplest baseline — 1/N allocation to each stock, no cash."""
    env      = PortfolioEnv(test_data, sector)
    n_stocks = env.n_stocks
    weights  = np.ones(n_stocks + 1) / (n_stocks + 1)
    obs, _   = env.reset()

    pv_history     = [1.0]
    ret_history    = []
    weight_history = []
    done           = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(weights)
        done = terminated or truncated
        ret_history.append(info["daily_return"])
        pv_history.append(info["portfolio_value"])
        weight_history.append(info["weights"])

    esg_base = test_data[sector]["esg"]

    return {
        "method"       : "equal_weight",
        "sector"       : sector,
        "cum_return"   : round(cumulative_return(pv_history),                 4),
        "sharpe"       : round(sharpe_ratio(ret_history),                     4),
        "max_drawdown" : round(max_drawdown(pv_history),                      4),
        "esg_rate"     : round(esg_compliance_rate(weight_history, esg_base), 4),
        "pv_history"   : pv_history,
    }


def run_all_baselines(test_data, train_data, obs_dim, n_stocks):
    print("\nRunning baselines...\n")
    print(f"  {'Method':<15} {'Sector':<12} {'Return':>8} {'Sharpe':>8} "
          f"{'Drawdown':>10} {'ESG%':>7}")
    print("-" * 65)

    all_results = []
    sectors     = list(test_data.keys())

    for sector in sectors:
        for fn, kwargs in [
            (run_markowitz,   {"test_data": test_data, "train_data": train_data, "sector": sector}),
            (run_vanilla_rl,  {"test_data": test_data, "sector": sector, "obs_dim": obs_dim, "n_stocks": n_stocks}),
            (run_equal_weight,{"test_data": test_data, "sector": sector}),
        ]:
            r = fn(**kwargs)
            all_results.append(r)
            print(f"  {r['method']:<15} {sector:<12} "
                  f"{r['cum_return']:>8.2%} {r['sharpe']:>8.3f} "
                  f"{r['max_drawdown']:>10.2%} {r['esg_rate']:>7.2%}")

    return all_results


if __name__ == "__main__":
    with open("data/processed/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("data/processed/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    sectors  = list(test_data.keys())
    n_stocks = test_data[sectors[0]]["X"].shape[2]
    obs_dim  = 406

    results = run_all_baselines(test_data, train_data, obs_dim, n_stocks)
    print("\nbaselines.py OK")