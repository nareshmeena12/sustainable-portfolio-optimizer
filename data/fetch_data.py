import os
import time
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
RAW_DIR    = "data/raw"

TICKERS = {
    "tech": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "energy": ["RELIANCE.NS", "ONGC.NS", "TATAPOWER.NS", "ADANIGREEN.NS"],
    "healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
}

# fallback ESG scores (0=worst, 1=best) based on MSCI/Sustainalytics public ratings
FALLBACK_ESG = {
    "TCS.NS"        : {"total": 0.75, "environmental": 0.72, "social": 0.78, "governance": 0.75},
    "INFY.NS"       : {"total": 0.77, "environmental": 0.75, "social": 0.80, "governance": 0.76},
    "WIPRO.NS"      : {"total": 0.73, "environmental": 0.70, "social": 0.75, "governance": 0.74},
    "HCLTECH.NS"    : {"total": 0.72, "environmental": 0.68, "social": 0.74, "governance": 0.74},
    "RELIANCE.NS"   : {"total": 0.52, "environmental": 0.38, "social": 0.55, "governance": 0.63},
    "ONGC.NS"       : {"total": 0.32, "environmental": 0.20, "social": 0.38, "governance": 0.38},
    "TATAPOWER.NS"  : {"total": 0.65, "environmental": 0.72, "social": 0.62, "governance": 0.61},
    "ADANIGREEN.NS" : {"total": 0.74, "environmental": 0.85, "social": 0.65, "governance": 0.72},
    "SUNPHARMA.NS"  : {"total": 0.62, "environmental": 0.58, "social": 0.64, "governance": 0.64},
    "DRREDDY.NS"    : {"total": 0.68, "environmental": 0.63, "social": 0.70, "governance": 0.71},
    "CIPLA.NS"      : {"total": 0.66, "environmental": 0.61, "social": 0.68, "governance": 0.69},
    "DIVISLAB.NS"   : {"total": 0.60, "environmental": 0.55, "social": 0.62, "governance": 0.63},
}

def get_all_tickers():
    return [t for sector in TICKERS.values() for t in sector]


def fetch_prices(tickers):
    print(f"\nDownloading prices for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)

    if data.empty:
        raise RuntimeError("Price download failed. Check your internet or ticker symbols.")

    close = data["Close"][tickers].ffill().dropna()
    print(f"Got {len(close)} trading days | {close.index[0].date()} to {close.index[-1].date()}")
    return data, close


def fetch_esg(tickers):
    print("\nFetching ESG scores...")
    records = []

    for ticker in tickers:
        try:
            sus = yf.Ticker(ticker).sustainability
            if sus is None or sus.empty:
                raise ValueError("empty")

            # yahoo returns risk scores (lower=better), we flip to 0-1 (higher=better)
            normalize = lambda v: round(1 - float(v) / 50, 3) if v is not None else None

            total = normalize(sus.loc["totalEsg"].values[0]         if "totalEsg"         in sus.index else None)
            env   = normalize(sus.loc["environmentScore"].values[0] if "environmentScore" in sus.index else None)
            soc   = normalize(sus.loc["socialScore"].values[0]      if "socialScore"      in sus.index else None)
            gov   = normalize(sus.loc["governanceScore"].values[0]  if "governanceScore"  in sus.index else None)

            if None in [total, env, soc, gov] or min(total, env, soc, gov) < 0:
                raise ValueError("incomplete scores")

            records.append({"ticker": ticker, "total": total, "environmental": env,
                            "social": soc, "governance": gov, "source": "yahoo"})
            print(f"  {ticker:<20} ESG={total:.2f}  [yahoo]")

        except Exception:
            fb = FALLBACK_ESG[ticker]
            records.append({**{"ticker": ticker, "source": "fallback"}, **fb})
            print(f"  {ticker:<20} ESG={fb['total']:.2f}  [fallback]")

        time.sleep(0.5)

    return pd.DataFrame(records).set_index("ticker")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    tickers = get_all_tickers()

    data, close = fetch_prices(tickers)
    esg_df      = fetch_esg(tickers)

    # sector map — needed by environment to split agents
    sector_map = pd.DataFrame(
        [{"ticker": t, "sector": s} for s, ts in TICKERS.items() for t in ts]
    ).set_index("ticker")

    data.to_csv(f"{RAW_DIR}/prices_ohlcv.csv")
    close.to_csv(f"{RAW_DIR}/prices_close.csv")
    esg_df.to_csv(f"{RAW_DIR}/esg_scores.csv")
    sector_map.to_csv(f"{RAW_DIR}/sector_map.csv")

    print(f"\nAll files saved to {RAW_DIR}/")
    print("Next: run preprocess.py")


if __name__ == "__main__":
    main()