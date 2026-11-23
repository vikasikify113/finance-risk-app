import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "NIFTY50_demo_sample_5minute.csv"
DATA_PATH = SAMPLE_PATH

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import load_csv

st.set_page_config(page_title="Dataset analysis", layout="wide")
st.title("Dataset analysis: 5-minute intraday data")
st.caption("Five recent Nifty 50 sessions from Yahoo Finance. This small sample demonstrates the file format; it is not a long-history training dataset.")


@st.cache_data
def load():
    return load_csv(DATA_PATH)


try:
    df = load()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Once the CSV is in data/, run `python train.py` from this project or with its full path.")
    st.stop()
c = st.columns(4)
c[0].metric("5-minute bars", f"{len(df):,}")
c[1].metric("Trading days", df.index.normalize().nunique())
c[2].metric("From", str(df.index.min().date()))
c[3].metric("To", str(df.index.max().date()))

tab1, tab2, tab3 = st.tabs(["Prices", "Intraday pattern", "Model results"])

with tab1:
    st.subheader("Index level (last bar of each day)")
    st.line_chart(df["close"].resample("D").last().dropna())
    st.dataframe(df.tail(20))

with tab2:
    d = df.copy()
    d["ret"] = d["close"].pct_change() * 100
    d["day"] = d.index.normalize()
    d.loc[d["day"] != d["day"].shift(1), "ret"] = np.nan
    d["time"] = d.index.strftime("%H:%M")
    by_time = d.groupby("time")["ret"].apply(lambda s: s.abs().mean())
    st.subheader("Average 5-minute move by time of day (%)")
    st.line_chart(by_time)
    st.markdown(
        "- Activity is highest right after the open and falls through the morning.\n"
        "- There is a smaller, sharp spike near 15:00. The median at that bar is also about double its neighbours, "
        "so it happens on a typical day, not only on a few extreme days. The cause is not established from this data."
    )

with tab3:
    results_path = ROOT / "models" / "walk_forward_results.csv"
    model_path = ROOT / "models" / "risk_model.joblib"
    if not results_path.is_file() or not model_path.is_file():
        st.info("Model results are not available yet. Run `python train.py` from the project folder first.")
    else:
        res = pd.read_csv(results_path, index_col=0)
        avg = res.mean()
        st.subheader("Walk-forward validation (5 folds, lower error is better)")
        st.dataframe(res.round(4))
        st.bar_chart(avg)
        naive = avg["Naive (last 30 min)"]
        for m in ["Ridge", "Random Forest", "Gradient Boosting"]:
            st.write(f"**{m}**: {100 * (1 - avg[m] / naive):.1f}% lower error than the naive baseline")
        bundle = joblib.load(model_path)
        st.subheader("What the final model relies on (standardised Ridge coefficients)")
        st.bar_chart(bundle["coef"])
        st.caption(f"Final model trained on {bundle['n_rows']:,} bars, {bundle['trained_from'][:10]} to {bundle['trained_to'][:10]}.")
