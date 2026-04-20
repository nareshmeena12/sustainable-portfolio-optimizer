import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")


class TransformerActor(nn.Module):
    """
    Transformer-based actor for portfolio management.
    
    Takes the last WINDOW days of market features as a sequence,
    uses attention to find the most relevant days, then outputs
    portfolio weights via a softmax head.
    """

    def __init__(self, n_stocks, n_features, window, hidden_dim=64,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        self.n_stocks  = n_stocks
        self.n_features = n_features
        self.window    = window

        # input dim per timestep = n_stocks * n_features (all stocks flattened per day)
        input_dim = n_stocks * n_features

        # project raw features to transformer embedding dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # learnable positional encoding — one vector per timestep
        self.pos_encoding = nn.Embedding(window, hidden_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = hidden_dim,
            nhead      = n_heads,
            dim_feedforward = hidden_dim * 4,
            dropout    = dropout,
            batch_first = True,   # (batch, seq, dim)
            activation = "gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # actor head — takes [CLS] token output → portfolio weights
        # +1 for cash position
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_stocks + 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        for m in self.actor_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """
        obs : (batch, obs_dim) — raw flattened observation from environment
        
        We extract just the window features from obs and ignore
        the appended weights and portfolio value (those are for the critic).
        """
        batch = obs.shape[0]

        # extract window features — first window*n_stocks*n_features elements
        seq_len   = self.window
        feat_dim  = self.n_stocks * self.n_features
        seq_flat  = obs[:, :seq_len * feat_dim]                 # (batch, window*feat)
        seq       = seq_flat.view(batch, seq_len, feat_dim)     # (batch, window, feat)

        # project to hidden dim
        x = self.input_proj(seq)                                # (batch, window, hidden)

        # add positional encoding
        positions = torch.arange(seq_len, device=obs.device)
        x = x + self.pos_encoding(positions).unsqueeze(0)      # (batch, window, hidden)

        # transformer — attends over time dimension
        x = self.transformer(x)                                 # (batch, window, hidden)

        # use mean pooling over time as the context vector
        context = x.mean(dim=1)                                 # (batch, hidden)

        # actor head → logits → softmax → weights
        logits  = self.actor_head(context)                      # (batch, n_stocks+1)
        weights = torch.softmax(logits, dim=-1)                 # sums to 1

        return weights


def build_actor(n_stocks, n_features, window, config=None):
    cfg = config or {}
    return TransformerActor(
        n_stocks    = n_stocks,
        n_features  = n_features,
        window      = window,
        hidden_dim  = cfg.get("hidden_dim", 64),
        n_heads     = cfg.get("n_heads",    4),
        n_layers    = cfg.get("n_layers",   2),
        dropout     = cfg.get("dropout",    0.1),
    )


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # match our environment exactly
    n_stocks   = 4
    n_features = 5
    window     = 20
    obs_dim    = 406
    batch      = 16

    actor = build_actor(n_stocks, n_features, window).to(device)
    obs   = torch.randn(batch, obs_dim).to(device)

    weights = actor(obs)

    print(f"Input shape  : {obs.shape}")
    print(f"Output shape : {weights.shape}")
    print(f"Weights sum  : {weights.sum(dim=-1).mean().item():.4f}  (should be 1.0)")
    print(f"Weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    print(f"Actor params : {sum(p.numel() for p in actor.parameters()):,}")
    print("Transformer actor OK")