import torch
import numpy as np
import sys
sys.path.append(".")

from models.transformer_actor import build_actor
from models.ppo import PPOUpdater, RolloutBuffer

class PortfolioAgent:
    """
    Single portfolio agent managing one sector.
    Wraps the Transformer actor, PINN critic, rollout buffer, and PPO updater.
    """

    def __init__(self, sector, n_stocks, n_features, window, obs_dim, pinn, hjb_loss_fn, config=None):
        self.sector  = sector
        self.obs_dim = obs_dim
        self.device  = config.get("device", torch.device("cpu")) if config else torch.device("cpu")

        # actor is sector-specific — each agent has its own weights
        self.actor = build_actor(n_stocks, n_features, window, config).to(self.device)

        # pinn and hjb_loss are shared across agents (passed in from agent_manager)
        self.pinn       = pinn
        self.hjb_loss   = hjb_loss_fn

        # PPO updater ties everything together
        self.updater = PPOUpdater(self.actor, self.pinn, self.hjb_loss, config)
        self.buffer  = RolloutBuffer()

        # episode tracking
        self.episode_rewards    = []
        self.episode_esg_scores = []
        self.portfolio_values   = []

        print(f"  Agent [{sector}] initialized | obs_dim={obs_dim} | device={self.device}")

    def select_action(self, obs, t_norm):
        """Sample action and get value estimate for current state."""
        action, log_prob = self.updater.get_action(obs)
        value            = self.updater.get_value(obs, t_norm)
        return action, log_prob, value

    def store(self, obs, action, reward, next_obs, log_prob, value, done, t_norm):
        """Store one transition in the rollout buffer."""
        self.buffer.add(obs, action, reward, next_obs, log_prob, value, done, t_norm)

    def update(self):
        """Run PPO update on collected buffer. Returns loss stats."""
        if len(self.buffer) == 0:
            return {}
        return self.updater.update(self.buffer)

    def track(self, reward, info):
        """Track episode stats for logging."""
        self.episode_rewards.append(reward)
        self.episode_esg_scores.append(
            np.dot(info["weights"][:-1], info["esg_scores"])
        )
        self.portfolio_values.append(info["portfolio_value"])

    def episode_summary(self):
        """Returns dict of episode-level metrics."""
        rewards = self.episode_rewards
        esg     = self.episode_esg_scores
        pv      = self.portfolio_values

        total_return = (pv[-1] / pv[0] - 1) if len(pv) > 1 else 0.0
        sharpe       = self._sharpe(rewards)
        esg_rate     = np.mean([1 if e >= 0.6 else 0 for e in esg])

        return {
            "sector"       : self.sector,
            "total_return" : round(total_return, 4),
            "sharpe"       : round(sharpe, 4),
            "esg_rate"     : round(esg_rate, 4),
            "mean_reward"  : round(np.mean(rewards), 6),
            "final_value"  : round(pv[-1], 4) if pv else 0.0,
        }

    def reset_tracking(self):
        self.episode_rewards    = []
        self.episode_esg_scores = []
        self.portfolio_values   = []

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def _sharpe(self, rewards, risk_free=0.04 / 252):
     r = np.array(rewards)
    # clip extreme values before computing sharpe
     r = np.clip(r, -0.1, 0.1)
     if r.std() < 1e-8:
        return 0.0
     return float((r.mean() - risk_free) / r.std() * np.sqrt(252))


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pinn.pinn_model import build_pinn, HJBLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    n_stocks, n_features, window, obs_dim = 4, 5, 20, 406

    config = {
        "device"    : device,
        "n_epochs"  : 2,
        "batch_size": 32,
        "lr_actor"  : 3e-4,
        "lr_critic" : 1e-3,
    }

    # shared PINN and HJB loss (in real training these come from agent_manager)
    pinn       = build_pinn(obs_dim).to(device)
    hjb_loss   = HJBLoss()

    agent = PortfolioAgent(
        sector      = "tech",
        n_stocks    = n_stocks,
        n_features  = n_features,
        window      = window,
        obs_dim     = obs_dim,
        pinn        = pinn,
        hjb_loss_fn = hjb_loss,
        config      = config,
    )

    # simulate one short episode
    for step in range(60):
        obs      = np.random.randn(obs_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        t_norm   = step / 252.0

        action, log_prob, value = agent.select_action(obs, t_norm)

        reward = np.random.randn() * 0.01
        done   = step == 59

        fake_info = {
            "weights"        : action,
            "esg_scores"     : np.random.uniform(0.4, 0.9, n_stocks),
            "portfolio_value": 1.0 + reward,
        }

        agent.store(obs, action, reward, next_obs, log_prob, value, float(done), t_norm)
        agent.track(reward, fake_info)

    stats   = agent.update()
    summary = agent.episode_summary()

    print(f"PPO stats : {stats}")
    print(f"Episode   : {summary}")
    print("Agent OK")