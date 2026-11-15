import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Financial Risk & Market Prediction", layout="wide")

TICKERS = ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS",
           "SBIN.NS", "AXISBANK.NS", "RELIANCE.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
           "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS"]
FEATURES = ["Vol5", "Volatility", "Vol20", "AbsReturn", "MA20_ratio", "RSI"]
RISK_FREE = 0.06


@st.cache_data(show_spinner="Downloading prices...")
def load_prices(ticker):
    df = yf.download(ticker, start="2020-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def prepare(df):
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Volatility"] = d["Return"].rolling(10).std()
    d["Vol5"] = d["Return"].rolling(5).std()
    d["Vol20"] = d["Return"].rolling(20).std()
    d["AbsReturn"] = d["Return"].abs()
    d["MA20_ratio"] = d["Close"].rolling(20).mean() / d["Close"] - 1
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    d["FutureVol"] = d["Return"].rolling(5).std().shift(-5)   # target: next 5 days' risk
    return d


def make_model():
    return RandomForestRegressor(n_estimators=200, max_depth=5,
                                 min_samples_leaf=20, random_state=42)


def risk_metrics(r):
    var95 = np.percentile(r, 5)
    cum = (1 + r).cumprod()
    return {"vol": r.std() * np.sqrt(252), "var": var95,
            "cvar": r[r <= var95].mean(),
            "dd": (cum / cum.cummax() - 1).min()}


def summarize(r):
    cum = (1 + r).cumprod()
    vol = r.std() * np.sqrt(252)
    return {"Total return": cum.iloc[-1] - 1, "Ann. volatility": vol,
            "Sharpe": (r.mean() * 252 - RISK_FREE) / vol,
            "Max drawdown": (cum / cum.cummax() - 1).min()}


@st.cache_data(show_spinner="Training model...")
def forecast(ticker):
    d = prepare(load_prices(ticker))
    train = d.dropna(subset=FEATURES + ["FutureVol"])
    model = make_model().fit(train[FEATURES], train["FutureVol"])
    latest = d[FEATURES].dropna().iloc[[-1]]
    pred = model.predict(latest)[0]
    pct = float((model.predict(train[FEATURES]) < pred).mean())
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    return pred, pct, imp


@st.cache_data(show_spinner="Running walk-forward backtest...")
def backtest(ticker, q):
    d = prepare(load_prices(ticker)).dropna(subset=FEATURES + ["FutureVol"])
    X, y = d[FEATURES], d["FutureVol"]
    flags = pd.Series(np.nan, index=d.index)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        thr = np.quantile(m.predict(X.iloc[tr]), q)
        flags.iloc[te] = (m.predict(X.iloc[te]) < thr).astype(float)
    t = d.loc[flags.dropna().index].copy()
    t["Held"] = flags.dropna().shift(1).fillna(1)
    trades = t["Held"].diff().abs().fillna(0)
    t["Strategy"] = t["Held"] * t["Return"] - trades * 0.001
    t["Buy & Hold"] = t["Return"]
    table = pd.DataFrame({"Buy & Hold": summarize(t["Buy & Hold"]),
                          "ML risk-aware": summarize(t["Strategy"])}).T
    table["Time invested"] = [1.0, t["Held"].mean()]
    growth = (1 + t[["Buy & Hold", "Strategy"]]).cumprod()
    return growth, table


# ---------- Page ----------
st.title("AI-Powered Financial Risk and Market Prediction System")
st.caption("Educational project, not financial advice.")

ticker = st.sidebar.selectbox("Choose a stock", TICKERS)
q = st.sidebar.slider("Go to cash when predicted risk is above this percentile",
                      0.50, 0.90, 0.70, 0.05)

df = load_prices(ticker)
if df.empty:
    st.error("Could not download data. Check your internet connection.")
    st.stop()
r = df["Close"].pct_change().dropna()

tab1, tab2, tab3 = st.tabs(["Price & risk", "Volatility forecast", "Backtest"])

with tab1:
    st.subheader(f"{ticker} closing price")
    st.line_chart(df["Close"])
    m = risk_metrics(r)
    c = st.columns(4)
    c[0].metric("Annual volatility", f"{m['vol']:.1%}")
    c[1].metric("VaR 95% (1 day)", f"{m['var']:.2%}")
    c[2].metric("CVaR 95%", f"{m['cvar']:.2%}")
    c[3].metric("Max drawdown", f"{m['dd']:.1%}")

with tab2:
    pred, pct, imp = forecast(ticker)
    level = "High" if pct > 0.70 else "Low" if pct < 0.30 else "Medium"
    c1, c2 = st.columns(2)
    c1.metric("Predicted next-5-day volatility (annualised)", f"{pred * np.sqrt(252):.1%}")
    c2.metric("Predicted risk level", level)
    st.write(f"The model expects higher risk than on {pct:.0%} of days in this stock's history.")
    st.subheader("Which inputs the model relies on")
    st.bar_chart(imp)

with tab3:
    growth, table = backtest(ticker, q)
    st.subheader("Growth of 1 rupee: Buy & Hold vs ML risk-aware")
    st.line_chart(growth)
    st.dataframe(table.round(3))
    st.caption("Out-of-sample, walk-forward, 0.1% cost per switch. Past results do not predict the future.")