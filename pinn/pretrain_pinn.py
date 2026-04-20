import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append(".")

from pinn.pinn_model import build_pinn, HJBLoss

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 1000
BATCH_SIZE = 256
LR         = 1e-3
GAMMA      = 0.99
ALPHA      = 0.1    # HJB loss weight
SAVE_PATH  = "checkpoints/pinn_pretrained.pt"

os.makedirs("checkpoints", exist_ok=True)


def load_data():
    with open("data/processed/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    return train_data


def generate_collocation_points(train_data, n_points=10000):
    """
    Build synthetic (state, t, reward, next_state, next_t, done) tuples
    from historical data for PINN pretraining.

    We use real returns and ESG scores to compute rewards so the PINN
    learns value estimates grounded in actual market dynamics.
    """
    states, next_states, rewards, ts, next_ts, dones = [], [], [], [], [], []

    sectors = list(train_data.keys())

    for _ in range(n_points):
        sector = np.random.choice(sectors)
        data   = train_data[sector]
        T      = len(data["X"])

        # random timestep
        idx = np.random.randint(0, T - 1)

        # state = flattened window features + uniform weights + portfolio value
        n_stocks = data["X"].shape[2]
        weights  = np.ones(n_stocks + 1) / (n_stocks + 1)
        pv       = np.random.uniform(0.8, 1.2)   # random portfolio value

        state      = np.concatenate([data["X"][idx].flatten(), weights, [pv]])
        next_state = np.concatenate([data["X"][idx + 1].flatten(), weights, [pv]])

        # reward = weighted return - ESG penalty
        log_ret    = data["y"][idx]
        esg        = data["esg"]
        stock_w    = weights[:-1]
        daily_ret  = np.dot(stock_w, log_ret)
        esg_pen    = np.dot(stock_w, 1.0 - esg)
        reward     = daily_ret - ALPHA * esg_pen

        # normalized time
        t      = idx / T
        next_t = (idx + 1) / T
        done   = float(idx == T - 2)

        states.append(state)
        next_states.append(next_state)
        rewards.append([reward])
        ts.append([t])
        next_ts.append([next_t])
        dones.append([done])

    def to_tensor(x):
        return torch.tensor(np.array(x), dtype=torch.float32).to(DEVICE)

    return (
        to_tensor(states),
        to_tensor(next_states),
        to_tensor(rewards),
        to_tensor(ts),
        to_tensor(next_ts),
        to_tensor(dones),
    )


def pretrain(pinn, hjb_loss_fn, train_data):
    optimizer = optim.Adam(pinn.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    print(f"\nPretraining PINN for {EPOCHS} epochs on {DEVICE}...")
    print(f"{'Epoch':>6}  {'Total':>10}  {'Data':>10}  {'HJB':>10}")
    print("-" * 44)

    states, next_states, rewards, ts, next_ts, dones = generate_collocation_points(train_data)
    N = len(states)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        pinn.train()

        # shuffle
        idx = torch.randperm(N)
        states      = states[idx]
        next_states = next_states[idx]
        rewards     = rewards[idx]
        ts          = ts[idx]
        next_ts     = next_ts[idx]
        dones       = dones[idx]

        epoch_loss = epoch_data = epoch_hjb = 0.0
        n_batches  = 0

        for i in range(0, N, BATCH_SIZE):
            s  = states[i:i+BATCH_SIZE]
            ns = next_states[i:i+BATCH_SIZE]
            r  = rewards[i:i+BATCH_SIZE]
            t  = ts[i:i+BATCH_SIZE]
            nt = next_ts[i:i+BATCH_SIZE]
            d  = dones[i:i+BATCH_SIZE]

            optimizer.zero_grad()
            loss, l_data, l_hjb = hjb_loss_fn(pinn, s, t, r, ns, nt, d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_data += l_data
            epoch_hjb  += l_hjb
            n_batches  += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_data = epoch_data / n_batches
        avg_hjb  = epoch_hjb  / n_batches

        if epoch % 100 == 0:
            print(f"{epoch:>6}  {avg_loss:>10.4f}  {avg_data:>10.4f}  {avg_hjb:>10.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(pinn.state_dict(), SAVE_PATH)

    print(f"\nBest loss : {best_loss:.4f}")
    print(f"Saved to  : {SAVE_PATH}")


def main():
    print(f"Device : {DEVICE}")

    train_data = load_data()

    # obs dim must match environment
    sample_sector = list(train_data.keys())[0]
    sample_X      = train_data[sample_sector]["X"][0]
    n_stocks      = sample_X.shape[1]
    obs_dim       = sample_X.flatten().shape[0] + (n_stocks + 1) + 1

    print(f"Obs dim : {obs_dim}")

    pinn        = build_pinn(obs_dim).to(DEVICE)
    hjb_loss_fn = HJBLoss(gamma=GAMMA, alpha=ALPHA)

    pretrain(pinn, hjb_loss_fn, train_data)

    # quick validation
    pinn.eval()
    pinn.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    dummy_state = torch.randn(1, obs_dim).to(DEVICE)
    dummy_t     = torch.tensor([[0.5]]).to(DEVICE)
    with torch.no_grad():
        v = pinn(dummy_state, dummy_t)
    print(f"\nValidation — V(random_state, t=0.5) = {v.item():.4f}")
    print("Pretraining complete. Next: transformer_actor.py")


if __name__ == "__main__":
    main()