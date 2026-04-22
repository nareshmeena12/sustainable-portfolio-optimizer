import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Dirichlet
import sys
sys.path.append(".")


class RolloutBuffer:
    """Stores experience from one episode for PPO update."""

    def __init__(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.next_states = []
        self.log_probs   = []
        self.values      = []
        self.dones       = []
        self.timesteps   = []

    def add(self, state, action, reward, next_state, log_prob, value, done, t):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.timesteps.append(t)

    def clear(self):
        self.__init__()

    def to_tensors(self, device):
        return (
            torch.tensor(np.array(self.states),      dtype=torch.float32).to(device),
            torch.tensor(np.array(self.actions),     dtype=torch.float32).to(device),
            torch.tensor(np.array(self.rewards),     dtype=torch.float32).to(device),
            torch.tensor(np.array(self.next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.log_probs),   dtype=torch.float32).to(device),
            torch.tensor(np.array(self.values),      dtype=torch.float32).to(device),
            torch.tensor(np.array(self.dones),       dtype=torch.float32).to(device),
            torch.tensor(np.array(self.timesteps),   dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.states)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Reduces variance of policy gradient without introducing too much bias.
    """
    advantages = []
    gae        = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns    = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


class PPOUpdater:
    """
    PPO update for one agent with Transformer actor and PINN critic.
    Uses Dirichlet distribution for continuous portfolio weights —
    more appropriate than Gaussian for simplex-constrained actions.
    """

    def __init__(self, actor, pinn, hjb_loss_fn, config=None):
        self.actor      = actor
        self.pinn       = pinn
        self.hjb_loss   = hjb_loss_fn

        cfg = config or {}
        self.lr_actor   = cfg.get("lr_actor",   3e-4)
        self.lr_critic  = cfg.get("lr_critic",  1e-3)
        self.gamma      = cfg.get("gamma",       0.99)
        self.lam        = cfg.get("lam",         0.95)
        self.clip_eps   = cfg.get("clip_eps",    0.2)
        self.entropy_c  = cfg.get("entropy_c",   0.01)
        self.n_epochs   = cfg.get("n_epochs",    4)
        self.batch_size = cfg.get("batch_size",  64)
        self.device     = cfg.get("device",      torch.device("cpu"))

        self.actor_opt  = torch.optim.Adam(actor.parameters(), lr=self.lr_actor)
        self.critic_opt = torch.optim.Adam(pinn.parameters(),  lr=self.lr_critic)

    def get_action(self, obs_np):
        """
        Sample action from Dirichlet distribution parameterized by actor output.
        Returns action, log_prob (both numpy).
        """
        self.actor.eval()
        with torch.no_grad():
            obs     = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            alpha   = self.actor(obs).squeeze(0) * 10 + 1e-6  # concentration params > 0
            dist    = Dirichlet(alpha)
            action  = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu().item()

    def get_value(self, obs_np, t_norm):
        """Get value estimate from PINN critic."""
        self.pinn.eval()
        with torch.no_grad():
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            t   = torch.tensor([[t_norm]], dtype=torch.float32).to(self.device)
            v   = self.pinn(obs, t)
        return v.cpu().item()

    def update(self, buffer):
        """Run PPO update on collected rollout buffer."""
        if len(buffer) == 0:
            return {}

        states, actions, rewards, next_states, old_log_probs, values, dones, ts = \
            buffer.to_tensors(self.device)

        rewards_np = rewards.cpu().numpy().tolist()
        values_np  = values.cpu().numpy().tolist()
        dones_np   = dones.cpu().numpy().tolist()

        # compute advantages using GAE
        advantages, returns = compute_gae(rewards_np, values_np, dones_np,
                                          self.gamma, self.lam)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(states)
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            idx = torch.randperm(N)

            for i in range(0, N, self.batch_size):
                b = idx[i:i + self.batch_size]
                if len(b) < 2:
                    continue

                s   = states[b]
                a   = actions[b]
                adv = advantages[b]
                ret = returns[b]
                olp = old_log_probs[b]
                t_b = ts[b].unsqueeze(-1)
                ns  = next_states[b]
                d_b = dones[b].unsqueeze(-1)
                r_b = rewards[b].unsqueeze(-1)
                nt  = (t_b + 1.0 / 252).clamp(0, 1)

                # ── actor update ──────────────────────────────────────────
                self.actor.train()
                alpha    = self.actor(s) * 10 + 1e-6
                dist     = Dirichlet(alpha)
                a_safe = a.clamp(1e-4, 1 - 1e-4)
                a_safe = a_safe / a_safe.sum(dim=-1, keepdim=True)
                log_prob = dist.log_prob(a_safe)
                entropy  = dist.entropy()

                ratio    = torch.exp(log_prob - olp)
                clip     = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * adv, clip * adv).mean() \
                             - self.entropy_c * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # ── critic update (PINN + HJB) ────────────────────────────
                self.pinn.train()
                critic_loss, l_data, l_hjb = self.hjb_loss(
                    self.pinn, s, t_b, r_b, ns, nt, d_b
                )

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.pinn.parameters(), 0.5)
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        buffer.clear()

        return {
            "actor_loss" : np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy"    : np.mean(entropies),
        }


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.transformer_actor import build_actor
    from pinn.pinn_model import build_pinn, HJBLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_stocks, n_features, window, obs_dim = 4, 5, 20, 406

    actor   = build_actor(n_stocks, n_features, window).to(device)
    pinn    = build_pinn(obs_dim).to(device)
    hjb_fn  = HJBLoss()
    config  = {"device": device, "n_epochs": 2, "batch_size": 32}

    updater = PPOUpdater(actor, pinn, hjb_fn, config)
    buffer  = RolloutBuffer()

    # fill buffer with dummy data
    for step in range(64):
        obs      = np.random.randn(obs_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        t_norm   = step / 252.0

        action, log_prob = updater.get_action(obs)
        value            = updater.get_value(obs, t_norm)

        buffer.add(obs, action, np.random.randn(), next_obs,
                   log_prob, value, 0.0, t_norm)

    stats = updater.update(buffer)

    print(f"Actor loss  : {stats['actor_loss']:.4f}")
    print(f"Critic loss : {stats['critic_loss']:.4f}")
    print(f"Entropy     : {stats['entropy']:.4f}")
    print("PPO updater OK")