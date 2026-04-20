import torch
import torch.nn as nn
import numpy as np


class PINNValueNetwork(nn.Module):
    """
    PINN-based value function V(x, t).
    Takes portfolio state + time as input, outputs scalar value estimate.
    Physics loss enforces the HJB equation during training.
    """

    def __init__(self, state_dim, hidden_dim=128, n_layers=4):
        super().__init__()

        # build MLP with tanh activations (smooth, differentiable — needed for HJB gradients)
        layers = [nn.Linear(state_dim + 1, hidden_dim), nn.Tanh()]  # +1 for time input
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, t):
        # state : (batch, state_dim)
        # t     : (batch, 1)  normalized time in [0, 1]
        x = torch.cat([state, t], dim=-1)
        return self.net(x)  # (batch, 1)


class HJBLoss(nn.Module):
    """
    Computes the HJB residual loss for the PINN critic.

    HJB equation:
        dV/dt + max_a [ f(x,a) * grad_x(V) + r(x,a) ] = 0

    In practice we don't take the max (that's the agent's job).
    We instead enforce the Bellman consistency:
        dV/dt + f(x,a) * grad_x(V) + r = 0

    where r = observed reward (return - ESG penalty).
    """

    def __init__(self, gamma=0.99, alpha=0.1):
        super().__init__()
        self.gamma = gamma   # discount factor
        self.alpha = alpha   # weight of physics loss vs data loss

    def compute_hjb_residual(self, pinn, state, t, reward, next_state, next_t, done):
        """
        Computes HJB residual at given collocation points.

        Bellman form of HJB:
            V(x,t) = r + gamma * V(x', t+dt)   (if not terminal)
            V(x,t) = r                           (if terminal)

        Residual = V(x,t) - [r + gamma * V(x', t+dt) * (1 - done)]
        """
        state     = state.requires_grad_(True)
        t_input   = t.requires_grad_(True)

        V_current = pinn(state, t_input)          # (batch, 1)

        with torch.no_grad():
            V_next = pinn(next_state, next_t)     # (batch, 1)

        target = reward + self.gamma * V_next * (1.0 - done)
        residual = V_current - target             # should be ~0

        # dV/dt via autograd — this is the physics part
        dV_dt = torch.autograd.grad(
            outputs=V_current.sum(),
            inputs=t_input,
            create_graph=True,
        )[0]

        # HJB physics term: dV/dt + r + gamma*V_next*(1-done) - V_current ~ 0
        # simplified: dV/dt should be close to -(r + gamma*V_next - V_current)
        hjb_term = dV_dt + reward - V_current * (1 - self.gamma)

        return residual, hjb_term

    def forward(self, pinn, state, t, reward, next_state, next_t, done):
        residual, hjb_term = self.compute_hjb_residual(
            pinn, state, t, reward, next_state, next_t, done
        )

        # data loss : standard TD error (Bellman residual)
        loss_data = (residual ** 2).mean()

        # physics loss : HJB equation residual
        loss_hjb  = (hjb_term ** 2).mean()

        total = loss_data + self.alpha * loss_hjb
        return total, loss_data.item(), loss_hjb.item()


def build_pinn(obs_dim, config=None):
    cfg = config or {}
    return PINNValueNetwork(
        state_dim  = obs_dim,
        hidden_dim = cfg.get("hidden_dim", 128),
        n_layers   = cfg.get("n_layers",   4),
    )


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    obs_dim = 406   # matches our environment
    batch   = 32

    pinn    = build_pinn(obs_dim).to(device)
    hjb_loss = HJBLoss(gamma=0.99, alpha=0.1)

    # dummy batch
    state      = torch.randn(batch, obs_dim).to(device)
    next_state = torch.randn(batch, obs_dim).to(device)
    t          = torch.rand(batch, 1).to(device)
    next_t     = (t + 0.004).clamp(0, 1)           # ~1/252 time step
    reward     = torch.randn(batch, 1).to(device)
    done       = torch.zeros(batch, 1).to(device)

    loss, l_data, l_hjb = hjb_loss(pinn, state, t, reward, next_state, next_t, done)

    print(f"Total loss : {loss.item():.4f}")
    print(f"Data loss  : {l_data:.4f}")
    print(f"HJB  loss  : {l_hjb:.4f}")
    print(f"PINN params: {sum(p.numel() for p in pinn.parameters()):,}")
    print("PINN model OK")