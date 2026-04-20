import os
import torch
import numpy as np
import pickle
import sys
sys.path.append(".")

from marl.agent import PortfolioAgent
from pinn.pinn_model import build_pinn, HJBLoss
from env.portfolio_env import PortfolioEnv


class AgentManager:
    """
    Manages all 3 sector agents + shared PINN critic.
    This is the brain of the multi-agent system.
    """

    def __init__(self, train_data, config=None):
        self.config     = config or {}
        self.device     = self.config.get("device", torch.device("cpu"))
        self.train_data = train_data
        self.sectors    = list(train_data.keys())

        # figure out dimensions from data
        sample      = train_data[self.sectors[0]]
        self.n_stocks   = sample["X"].shape[2]
        self.n_features = sample["X"].shape[3]
        self.window     = sample["X"].shape[1]
        self.obs_dim    = (self.window * self.n_stocks * self.n_features
                          + self.n_stocks + 1 + 1)

        print(f"\nInitializing AgentManager")
        print(f"  Sectors  : {self.sectors}")
        print(f"  Obs dim  : {self.obs_dim}")
        print(f"  Device   : {self.device}")
        print(f"  Agents   : {len(self.sectors)}")

        # shared PINN critic — all agents use this
        self.pinn     = build_pinn(self.obs_dim, config).to(self.device)
        self.hjb_loss = HJBLoss(
            gamma = self.config.get("gamma", 0.99),
            alpha = self.config.get("pinn_alpha", 0.1),
        )

        # load pretrained PINN if available
        pinn_path = self.config.get("pinn_path", "checkpoints/pinn_pretrained.pt")
        if os.path.exists(pinn_path):
            self.pinn.load_state_dict(torch.load(pinn_path, map_location=self.device))
            print(f"  Loaded pretrained PINN from {pinn_path}")
        else:
            print(f"  No pretrained PINN found — starting from scratch")

        # one environment per sector
        self.envs = {
            sector: PortfolioEnv(train_data, sector, config)
            for sector in self.sectors
        }

        # one agent per sector — all share the same pinn
        self.agents = {
            sector: PortfolioAgent(
                sector      = sector,
                n_stocks    = self.n_stocks,
                n_features  = self.n_features,
                window      = self.window,
                obs_dim     = self.obs_dim,
                pinn        = self.pinn,
                hjb_loss_fn = self.hjb_loss,
                config      = self.config,
            )
            for sector in self.sectors
        }

        print(f"\nAll agents ready.\n")

    def run_episode(self):
        """
        Run one full episode across all sectors simultaneously.
        Each agent interacts with its own environment independently.
        Returns combined stats for the episode.
        """
        # reset everything
        obs_dict  = {}
        done_dict = {s: False for s in self.sectors}

        for sector in self.sectors:
            obs, _  = self.envs[sector].reset()
            obs_dict[sector] = obs
            self.agents[sector].reset_tracking()

        step    = 0
        max_steps = self.config.get("max_steps", 252)

        while not all(done_dict.values()) and step < max_steps:
            for sector in self.sectors:
                if done_dict[sector]:
                    continue

                agent = self.agents[sector]
                env   = self.envs[sector]
                obs   = obs_dict[sector]
                t_norm = step / max_steps

                # agent decides action
                action, log_prob, value = agent.select_action(obs, t_norm)

                # environment steps
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # store transition
                agent.store(obs, action, reward, next_obs,
                           log_prob, value, float(done), t_norm)
                agent.track(reward, info)

                obs_dict[sector]  = next_obs
                done_dict[sector] = done

            step += 1

        # update all agents after episode
        update_stats = {}
        for sector, agent in self.agents.items():
            stats = agent.update()
            update_stats[sector] = stats

        # collect episode summaries
        summaries = {
            sector: agent.episode_summary()
            for sector, agent in self.agents.items()
        }

        return summaries, update_stats

    def train(self, n_episodes):
        """Main training loop."""
        print(f"Starting training for {n_episodes} episodes...\n")
        print(f"{'Ep':>5} | {'Sector':<12} | {'Return':>8} | {'Sharpe':>8} | "
              f"{'ESG%':>6} | {'A_loss':>8} | {'C_loss':>8}")
        print("-" * 72)

        history = []

        for ep in range(1, n_episodes + 1):
            summaries, update_stats = self.run_episode()

            ep_log = {"episode": ep, "sectors": {}}

            for sector in self.sectors:
                s     = summaries[sector]
                stats = update_stats.get(sector, {})

                ep_log["sectors"][sector] = {**s, **stats}

                if ep % 10 == 0:
                    print(
                        f"{ep:>5} | {sector:<12} | "
                        f"{s['total_return']:>8.4f} | "
                        f"{s['sharpe']:>8.4f} | "
                        f"{s['esg_rate']:>6.2%} | "
                        f"{stats.get('actor_loss', 0):>8.4f} | "
                        f"{stats.get('critic_loss', 0):>8.4f}"
                    )

            history.append(ep_log)

            # save checkpoint every 50 episodes
            if ep % 50 == 0:
                self.save(f"checkpoints/ep_{ep}")
                print(f"\n  Checkpoint saved at episode {ep}\n")

        print("\nTraining complete.")
        return history

    def save(self, prefix):
        os.makedirs(prefix, exist_ok=True)
        # save each actor
        for sector, agent in self.agents.items():
            agent.save(f"{prefix}/actor_{sector}.pt")
        # save shared PINN
        torch.save(self.pinn.state_dict(), f"{prefix}/pinn.pt")

    def load(self, prefix):
        for sector, agent in self.agents.items():
            path = f"{prefix}/actor_{sector}.pt"
            if os.path.exists(path):
                agent.load(path)
        pinn_path = f"{prefix}/pinn.pt"
        if os.path.exists(pinn_path):
            self.pinn.load_state_dict(torch.load(pinn_path, map_location=self.device))
        print(f"Loaded checkpoint from {prefix}")


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("data/processed/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "device"     : device,
        "n_epochs"   : 2,
        "batch_size" : 32,
        "lr_actor"   : 3e-4,
        "lr_critic"  : 1e-3,
        "gamma"      : 0.99,
        "esg_lambda" : 0.3,
        "max_steps"  : 30,   # short for testing
    }

    manager = AgentManager(train_data, config)

    # run 3 episodes as a smoke test
    history = manager.train(n_episodes=3)
    print("\nAgentManager OK")