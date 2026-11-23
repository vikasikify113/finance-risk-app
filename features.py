import numpy as np
import pandas as pd
from pathlib import Path

BASE = ["rv6", "rv12", "abs_ret", "prev_day_vol", "minutes"]
FEATS = BASE + ["seasonal"]


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put NIFTY50_demo_sample_5minute.csv or "
            "NIFTYMS400_MQ100_5minute.csv in the project's data folder."
        )
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset {path.name} is missing required column(s): {', '.join(sorted(missing))}."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError(f"Dataset {path.name} contains no valid dates.")
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    prices = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(subset=["open", "high", "low", "close"])
    if prices.empty:
        raise ValueError(f"Dataset {path.name} contains no valid OHLC price rows.")
    return prices


def make_features(raw):
    m = raw.copy()
    m["ret"] = m["close"].pct_change() * 100
    m["day"] = m.index.normalize()
    m["time"] = m.index.strftime("%H:%M")
    m.loc[m["day"] != m["day"].shift(1), "ret"] = np.nan   # drop overnight gap
    m["sq"] = m["ret"] ** 2
    m["fut_rv"] = np.sqrt(m.groupby("day")["sq"].transform(
        lambda s: s.rolling(6).mean().shift(-6)))
    m["rv6"] = np.sqrt(m["sq"].rolling(6, min_periods=3).mean())
    m["rv12"] = np.sqrt(m["sq"].rolling(12, min_periods=6).mean())
    m["abs_ret"] = m["ret"].abs()
    daily_vol = m.groupby("day")["ret"].std()
    m["prev_day_vol"] = m["day"].map(daily_vol.shift(1))
    m["minutes"] = (m.index.hour - 9) * 60 + m.index.minute - 15
    return m
