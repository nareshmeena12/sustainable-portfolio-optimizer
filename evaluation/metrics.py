import numpy as np
import pickle
import torch
import sys
sys.path.append(".")


def sharpe_ratio(returns, risk_free=0.04/252):
    r = np.array(returns)
    if r.std() < 1e-8:
        return 0.0
    return float((r.mean() - risk_free) / r.std() * np.sqrt(252))


def max_drawdown(portfolio_values):
    pv   = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    dd   = (pv - peak) / peak
    return float(dd.min())


def cumulative_return(portfolio_values):
    pv = np.array(portfolio_values)
    return float(pv[-1] / pv[0] - 1)


def esg_compliance_rate(weights_history, esg_scores, threshold=0.6):
    compliant = 0
    for w in weights_history:
        stock_w = np.array(w[:-1])  # exclude cash
        score   = np.dot(stock_w, esg_scores)
        if score >= threshold:
            compliant += 1
    return compliant / len(weights_history)


def evaluate_agent(agent, env, test_data, sector, n_episodes=5):
    """
    Run agent on test environment for n_episodes and collect metrics.
    Returns averaged metrics across episodes.
    """
    from env.portfolio_env import PortfolioEnv

    test_env = PortfolioEnv(test_data, sector)
    esg_base = test_data[sector]["esg"]

    all_returns    = []
    all_sharpes    = []
    all_drawdowns  = []
    all_esg_rates  = []

    for ep in range(n_episodes):
        obs, _        = test_env.reset()
        done          = False
        step          = 0
        pv_history    = [1.0]
        ret_history   = []
        weight_history = []

        while not done:
            t_norm           = step / 252.0
            action, _, _     = agent.select_action(obs, t_norm)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done             = terminated or truncated

            ret_history.append(info["daily_return"])
            pv_history.append(info["portfolio_value"])
            weight_history.append(info["weights"])
            step += 1

        all_returns.append(cumulative_return(pv_history))
        all_sharpes.append(sharpe_ratio(ret_history))
        all_drawdowns.append(max_drawdown(pv_history))
        all_esg_rates.append(esg_compliance_rate(weight_history, esg_base))

    return {
        "sector"        : sector,
        "cum_return"    : round(np.mean(all_returns),   4),
        "sharpe"        : round(np.mean(all_sharpes),   4),
        "max_drawdown"  : round(np.mean(all_drawdowns), 4),
        "esg_rate"      : round(np.mean(all_esg_rates), 4),
    }


def run_full_evaluation(manager, test_data):
    print("\nEvaluating agents on test data (2023-2024)...\n")
    results = {}

    for sector, agent in manager.agents.items():
        r = evaluate_agent(agent, None, test_data, sector)
        results[sector] = r
        print(f"  {sector:<12} | return={r['cum_return']:>7.2%} | "
              f"sharpe={r['sharpe']:>6.3f} | "
              f"drawdown={r['max_drawdown']:>7.2%} | "
              f"esg={r['esg_rate']:>5.2%}")

    return results


if __name__ == "__main__":
    # quick standalone test with dummy data
    dummy_returns = np.random.randn(252) * 0.01
    dummy_pv      = np.cumprod(1 + dummy_returns)
    dummy_weights = [np.array([0.25, 0.25, 0.25, 0.15, 0.10])] * 100
    dummy_esg     = np.array([0.75, 0.52, 0.65, 0.74])

    print(f"Sharpe        : {sharpe_ratio(dummy_returns):.4f}")
    print(f"Max Drawdown  : {max_drawdown(dummy_pv):.4f}")
    print(f"Cum Return    : {cumulative_return(dummy_pv):.4f}")
    print(f"ESG Rate      : {esg_compliance_rate(dummy_weights, dummy_esg):.4f}")
    print("metrics.py OK")