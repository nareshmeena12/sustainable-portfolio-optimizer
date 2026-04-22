import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Single-agent portfolio environment for one sector (4 stocks).
    Observation : last WINDOW days of (n_stocks x n_features) + current weights + portfolio value
    Action      : portfolio weights over n_stocks + 1 cash position (sums to 1)
    Reward      : daily return - lambda * ESG_penalty
    """

    def __init__(self, data, sector, config=None):
        super().__init__()

        cfg = config or {}
        self.esg_lambda     = cfg.get("esg_lambda",    1.0)   # ESG penalty weight
        self.initial_value  = cfg.get("initial_value", 1.0)   # normalized portfolio start value
        self.esg_drift_std  = cfg.get("esg_drift_std", 0.002) # stochastic ESG noise per step
        self.max_drawdown   = cfg.get("max_drawdown",  0.5)   # episode ends if portfolio drops 50%
        self.transaction_cost = cfg.get("transaction_cost", 0.001)  # 0.1% per trade

        self.X       = data[sector]["X"]        # (T, WINDOW, n_stocks, n_features)
        self.y       = data[sector]["y"]        # (T, n_stocks) — next day log returns
        self.esg_base = data[sector]["esg"].copy()  # (n_stocks,) base ESG scores
        self.tickers = data[sector]["tickers"]

        self.T        = len(self.X)
        self.n_stocks = self.X.shape[2]
        self.n_feat   = self.X.shape[3]
        self.window   = self.X.shape[1]

        # action space : weights for n_stocks + 1 cash, all in [0,1], softmax applied inside step
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_stocks + 1,),
            dtype=np.float32
        )

        # obs = flattened window features + current weights + portfolio value (scalar)
        obs_dim = self.window * self.n_stocks * self.n_feat + (self.n_stocks + 1) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    # ── core methods ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # sample random start so agent sees diverse market conditions each episode
        self.t              = np.random.randint(0, self.T - 252) if self.T > 252 else 0
        self.portfolio_value = self.initial_value
        self.weights        = self._uniform_weights()
        self.esg_current    = self.esg_base.copy()  # ESG will drift stochastically
        self.done           = False

        return self._get_obs(), {}

    def step(self, action):
        assert not self.done, "Call reset() before stepping again"

        # softmax to get valid portfolio weights (sum to 1, all positive)
        new_weights = self._softmax(action)
        stock_weights = new_weights[:-1]   # first n_stocks weights
        cash_weight   = new_weights[-1]    # last element is cash

        # transaction cost — proportional to how much weights changed
        turnover = np.abs(new_weights - self.weights).sum() / 2.0
        cost     = self.transaction_cost * turnover

        # next day returns for this sector
        log_returns = self.y[self.t]                       # (n_stocks,)
        daily_ret   = np.dot(stock_weights, log_returns)   # weighted portfolio return

        # update portfolio value
        self.portfolio_value *= np.exp(daily_ret) * (1 - cost)

        # ESG stochastic drift — scores change slowly with small noise each day
        noise = np.random.normal(0, self.esg_drift_std, size=self.n_stocks)
        self.esg_current = np.clip(self.esg_base + noise, 0.0, 1.0)

        # ESG penalty — weighted sum of (1 - esg_score) for stock holdings
        weighted_esg = np.dot(stock_weights, self.esg_current)
        esg_penalty  = max(0.0, 0.6 - weighted_esg)  # only penalize if below threshold

        # reward
        reward = daily_ret * 100 - self.esg_lambda * esg_penalty - cost * 100


        # update state
        self.weights = new_weights
        self.t      += 1

        # episode termination conditions
        truncated = self.t >= self.T - 1
        terminated = self.portfolio_value < self.initial_value * (1 - self.max_drawdown)
        self.done  = truncated or terminated

        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return"   : daily_ret,
            "esg_penalty"    : esg_penalty,
            "esg_scores"     : self.esg_current.copy(),
            "weights"        : new_weights.copy(),
            "turnover"       : turnover,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self):
        # flatten window features (WINDOW x n_stocks x n_feat)
        window_feat = self.X[self.t].flatten()

        # append current weights and portfolio value
        obs = np.concatenate([
            window_feat,
            self.weights,
            [self.portfolio_value]
        ]).astype(np.float32)

        return obs

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def _uniform_weights(self):
        # start with equal allocation across stocks + small cash position
        n = self.n_stocks + 1
        return np.ones(n, dtype=np.float32) / n

    def get_esg_compliance(self, threshold=0.6):
        # helper: is current portfolio ESG-compliant?
        stock_weights = self.weights[:-1]
        weighted_esg  = np.dot(stock_weights, self.esg_current)
        return weighted_esg >= threshold

    def render(self):
        stock_weights = self.weights[:-1]
        print(
            f"t={self.t:4d} | "
            f"value={self.portfolio_value:.4f} | "
            f"esg={np.dot(stock_weights, self.esg_current):.3f} | "
            f"weights={np.round(self.weights, 3)}"
        )