import os
import numpy as np
import pandas as pd
import pickle

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

WINDOW_SIZE = 20  # transformer lookback window

TICKERS = {
    "tech"      : ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "energy"    : ["RELIANCE.NS", "ONGC.NS", "TATAPOWER.NS", "ADANIGREEN.NS"],
    "healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
}


def load_raw():
    close     = pd.read_csv(f"{RAW_DIR}/prices_close.csv", index_col=0, parse_dates=True)
    esg_df    = pd.read_csv(f"{RAW_DIR}/esg_scores.csv",  index_col=0)
    sector_map = pd.read_csv(f"{RAW_DIR}/sector_map.csv", index_col=0)
    print(f"Loaded prices : {close.shape}  ({close.index[0].date()} to {close.index[-1].date()})")
    print(f"Loaded ESG    : {esg_df.shape}")
    return close, esg_df, sector_map


def compute_features(close):
    """
    For each stock compute:
      - log_return         : daily log return
      - rolling_mean       : 20-day rolling mean of log returns
      - rolling_vol        : 20-day rolling std  of log returns (volatility)
      - volume_proxy       : we don't have volume separately so we use abs(log_return) as proxy
    Returns a dict: ticker -> DataFrame with these 4 columns
    """
    features = {}
    log_ret = np.log(close / close.shift(1)).dropna()

    for ticker in close.columns:
        r = log_ret[ticker]
        df = pd.DataFrame(index=r.index)
        df["log_return"]   = r
        df["rolling_mean"] = r.rolling(WINDOW_SIZE).mean()
        df["rolling_vol"]  = r.rolling(WINDOW_SIZE).std()
        df["abs_return"]   = r.abs()   # volume proxy
        df = df.dropna()
        features[ticker] = df

    print(f"Features computed for {len(features)} tickers | window={WINDOW_SIZE}")
    return features, log_ret.dropna()


def normalize_features(features, train_end):
    """
    Normalize each feature using train-period mean and std only.
    This prevents data leakage into the test set.
    """
    normalized = {}
    stats = {}  # save stats so we can inverse-transform later if needed

    for ticker, df in features.items():
        train_df = df[df.index <= train_end]
        mean = train_df.mean()
        std  = train_df.std().replace(0, 1)  # avoid division by zero

        norm_df = (df - mean) / std
        normalized[ticker] = norm_df
        stats[ticker] = {"mean": mean, "std": std}

    print(f"Normalized features using train period stats (no data leakage)")
    return normalized, stats


def attach_esg(normalized, esg_df):
    """
    Add ESG total score as a constant feature column for each ticker.
    ESG is already on 0-1 scale from fetch_data.py.
    """
    for ticker, df in normalized.items():
        esg_score = esg_df.loc[ticker, "total"] if ticker in esg_df.index else 0.5
        df["esg_score"] = esg_score  # constant column, will drift in environment

    print(f"ESG scores attached to all tickers")
    return normalized


def build_sequences(normalized, log_ret, esg_df):
    """
    For each sector build (X, y) sequences for the environment and transformer.

    X shape : (N, WINDOW_SIZE, n_stocks, n_features)
    y shape : (N, n_stocks)  — next day log returns for all stocks in sector

    This is what the environment will replay during training.
    """
    all_tickers = [t for ts in TICKERS.values() for t in ts]

    # align all tickers to common dates
    common_dates = normalized[all_tickers[0]].index
    for ticker in all_tickers[1:]:
        common_dates = common_dates.intersection(normalized[ticker].index)

    print(f"Common trading dates across all tickers: {len(common_dates)}")

    sector_data = {}
    n_features  = normalized[all_tickers[0]].shape[1]  # should be 5

    for sector, tickers in TICKERS.items():
        feat_arrays = []
        for ticker in tickers:
            arr = normalized[ticker].loc[common_dates].values  # (T, n_features)
            feat_arrays.append(arr)

        # stack -> (T, n_stocks, n_features)
        stacked = np.stack(feat_arrays, axis=1)

        # returns for this sector on common dates
        ret_arr = log_ret[tickers].loc[common_dates].values  # (T, n_stocks)

        # build sliding windows
        X, y, dates = [], [], []
        for i in range(WINDOW_SIZE, len(stacked)):
            X.append(stacked[i - WINDOW_SIZE:i])   # (WINDOW, n_stocks, n_features)
            y.append(ret_arr[i])                    # (n_stocks,)
            dates.append(common_dates[i])

        sector_data[sector] = {
            "X"      : np.array(X, dtype=np.float32),
            "y"      : np.array(y, dtype=np.float32),
            "dates"  : np.array(dates),
            "tickers": tickers,
            "esg"    : np.array([esg_df.loc[t, "total"] for t in tickers], dtype=np.float32),
        }

        print(f"  {sector:<12} X={sector_data[sector]['X'].shape}  y={sector_data[sector]['y'].shape}")

    return sector_data, common_dates


def train_test_split(sector_data):
    train_data, test_data = {}, {}

    for sector, data in sector_data.items():
        dates      = pd.to_datetime(data["dates"])
        train_mask = dates <= TRAIN_END
        test_mask  = dates > TRAIN_END

        train_data[sector] = {
            "X"      : data["X"][train_mask],
            "y"      : data["y"][train_mask],
            "dates"  : data["dates"][train_mask],
            "tickers": data["tickers"],
            "esg"    : data["esg"],
        }
        test_data[sector] = {
            "X"      : data["X"][test_mask],
            "y"      : data["y"][test_mask],
            "dates"  : data["dates"][test_mask],
            "tickers": data["tickers"],
            "esg"    : data["esg"],
        }

        print(
            f"  {sector:<12} "
            f"train={train_data[sector]['X'].shape[0]} days | "
            f"test={test_data[sector]['X'].shape[0]} days"
        )

    return train_data, test_data


def save(train_data, test_data, stats):
    os.makedirs(PROC_DIR, exist_ok=True)

    with open(f"{PROC_DIR}/train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(f"{PROC_DIR}/test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    with open(f"{PROC_DIR}/norm_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print(f"\nSaved to {PROC_DIR}/")
    print("  train_data.pkl")
    print("  test_data.pkl")
    print("  norm_stats.pkl")


def sanity_check(train_data, test_data):
    print("\nSanity check:")
    for sector in TICKERS:
        tr = train_data[sector]
        te = test_data[sector]
        assert not np.isnan(tr["X"]).any(), f"NaN in train X for {sector}"
        assert not np.isnan(te["X"]).any(), f"NaN in test X for {sector}"
        assert not np.isnan(tr["y"]).any(), f"NaN in train y for {sector}"
        print(f"  {sector:<12} OK — X range [{tr['X'].min():.2f}, {tr['X'].max():.2f}]")


def main():
    print("=" * 50)
    print("  PREPROCESSING")
    print("=" * 50)

    close, esg_df, sector_map = load_raw()

    print("\nComputing features...")
    features, log_ret = compute_features(close)

    print("\nNormalizing...")
    normalized, stats = normalize_features(features, TRAIN_END)

    print("\nAttaching ESG scores...")
    normalized = attach_esg(normalized, esg_df)

    print("\nBuilding sequences...")
    sector_data, _ = build_sequences(normalized, log_ret, esg_df)

    print("\nSplitting train/test...")
    train_data, test_data = train_test_split(sector_data)

    sanity_check(train_data, test_data)
    save(train_data, test_data, stats)

    print("\nDone. Next: build portfolio_env.py")


if __name__ == "__main__":
    main()