import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from stock_universe import (
    NSE_52_WEEK,
    NSE_CORPORATE_FILINGS,
    NSE_LIVE_MARKET,
    NSE_NIFTY50,
    NSE_TOP_MOVERS,
    load_nifty50,
)

st.set_page_config(page_title="Live market analysis", layout="wide")
st.title("Live market analysis")
st.caption(
    "Five-minute prices from Yahoo Finance. Quotes can be delayed; this page refreshes every five minutes. "
    "Indicators describe recent price action and are not trading recommendations."
)
st.link_button(
    "Open NSE India live equity market",
    NSE_LIVE_MARKET,
)
with st.expander("More official NSE market tools"):
    st.link_button("Top gainers and losers", NSE_TOP_MOVERS)
    st.link_button("52-week highs and lows", NSE_52_WEEK)
    st.link_button("Company announcements", NSE_CORPORATE_FILINGS)

INDEX_SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
}
STOCKS = load_nifty50()


@st.cache_data(ttl=60, max_entries=10, show_spinner="Fetching five-minute market data…")
def get_intraday(symbol: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        period="60d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no five-minute prices for {symbol}.")

    # yfinance may return either simple columns or (price, ticker) MultiIndex columns.
    wanted = {"open", "high", "low", "close", "volume"}
    values = {}
    for column in raw.columns:
        parts = column if isinstance(column, tuple) else (column,)
        field = next((str(part).lower() for part in parts if str(part).lower() in wanted), None)
        if field is not None and field not in values:
            values[field] = raw[column]

    missing = {"open", "high", "low", "close"}.difference(values)
    if missing:
        raise ValueError(f"Yahoo Finance data is missing price fields: {', '.join(sorted(missing))}.")

    prices = pd.DataFrame(values, index=raw.index).dropna(subset=["open", "high", "low", "close"])
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("Asia/Kolkata").tz_localize(None)
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError(f"Yahoo Finance returned no usable five-minute prices for {symbol}.")
    return prices


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    average_gain = change.clip(lower=0).rolling(window, min_periods=window).mean()
    average_loss = -change.clip(upper=0).rolling(window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    return rsi


with st.sidebar:
    st.subheader("Market")
    market_type = st.radio("Choose a list", ["NIFTY 50 stocks", "Indices"])
    if market_type == "Indices":
        instrument = st.selectbox("Index", list(INDEX_SYMBOLS))
        symbol = INDEX_SYMBOLS[instrument]
    else:
        sector = st.selectbox("Sector", ["All sectors", *sorted(STOCKS["sector"].unique())])
        group = st.selectbox(
            "Company group",
            ["All company groups", *sorted(STOCKS["company_group"].unique())],
        )
        choices = STOCKS.copy()
        if sector != "All sectors":
            choices = choices[choices["sector"] == sector]
        if group != "All company groups":
            choices = choices[choices["company_group"] == group]
        if choices.empty:
            st.info("No stocks match both filters. Choose another sector or company group.")
            st.stop()
        selected_stock = st.selectbox(
            "Company",
            choices.to_dict("records"),
            format_func=lambda item: f"{item['company']} ({item['symbol']})",
        )
        instrument = f"{selected_stock['company']} ({selected_stock['symbol']})"
        symbol = selected_stock["yahoo_ticker"]
        st.caption("NIFTY 50 membership snapshot: 26 Nov 2026. Yahoo Finance prices may be delayed.")
        st.link_button("Open NSE NIFTY 50 constituents", NSE_NIFTY50)
    visible_range = st.selectbox("Chart range", ["1 day", "5 days", "1 month", "3 months"], index=1)
    if st.button("Refresh data now", icon=":material/refresh:"):
        get_intraday.clear()


@st.fragment(run_every=300)
def show_live_analysis(symbol: str, name: str, selected_range: str) -> None:
    try:
        raw = get_intraday(symbol)
    except Exception as exc:
        st.error(
            f"Could not load market data for {name}. Check your internet connection and try refreshing. "
            f"Details: {exc}",
            title="Market data unavailable",
        )
        return

    latest_time = raw.index[-1]
    last_session_day = latest_time.normalize()
    latest_session = raw.loc[raw.index.normalize() == last_session_day]
    daily_closes = raw.groupby(raw.index.normalize())["close"].last()
    previous_close = daily_closes.iloc[-2] if len(daily_closes) > 1 else np.nan
    latest_price = float(raw["close"].iloc[-1])
    session_change = latest_price / previous_close - 1 if pd.notna(previous_close) and previous_close else np.nan
    session_move = float((latest_session["close"].iloc[-1] / latest_session["close"].iloc[0] - 1) * 100)
    session_returns = latest_session["close"].pct_change().dropna()
    realized_vol = float(np.sqrt((session_returns ** 2).sum()) * 100) if not session_returns.empty else np.nan

    close = raw["close"]
    rsi = calculate_rsi(close)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    last_rsi = rsi.iloc[-1]
    last_ma20 = ma20.iloc[-1]
    last_ma50 = ma50.iloc[-1]

    if visible_range == "1 day":
        chart_data = latest_session
    else:
        sessions = {"5 days": 5, "1 month": 20, "3 months": 60}[visible_range]
        session_days = raw.index.normalize().unique()[-sessions:]
        chart_data = raw.loc[raw.index.normalize().isin(session_days)]

    info = (
        f"**{name}** (`{symbol}`) · Last bar: {latest_time:%d %b %Y, %H:%M} IST · "
        "Auto-refresh: every 5 minutes"
    )
    st.info(info)

    with st.container(horizontal=True):
        st.metric("Latest price", f"{latest_price:,.2f}",
                  f"{session_change:+.2%} vs previous session close" if pd.notna(session_change) else None)
        st.metric("Today's move", f"{session_move:+.2f}%")
        st.metric("Session realized volatility", f"{realized_vol:.2f}%" if pd.notna(realized_vol) else "—")
        st.metric("RSI (14 bars)", f"{last_rsi:.1f}" if pd.notna(last_rsi) else "Warming up")

    price_chart = chart_data[["close"]].copy()
    price_chart["20-bar average"] = ma20.reindex(chart_data.index)
    price_chart["50-bar average"] = ma50.reindex(chart_data.index)

    price_tab, indicators_tab, bars_tab = st.tabs(["Price and trend", "Indicators", "Recent bars"])
    with price_tab:
        st.line_chart(
            price_chart,
            alt=f"{name} five-minute closing price with 20-bar and 50-bar averages",
        )
        if "volume" in chart_data.columns and chart_data["volume"].notna().any():
            st.bar_chart(chart_data["volume"], alt=f"{name} five-minute trading volume")

    with indicators_tab:
        left, right = st.columns(2)
        with left:
            indicator_chart = pd.DataFrame({"RSI (14 bars)": rsi.reindex(chart_data.index)})
            st.line_chart(indicator_chart, alt=f"{name} 14-bar relative strength index")
        with right:
            st.metric("20-bar moving average", f"{last_ma20:,.2f}" if pd.notna(last_ma20) else "Warming up")
            st.metric("50-bar moving average", f"{last_ma50:,.2f}" if pd.notna(last_ma50) else "Warming up")
            if pd.notna(last_rsi):
                rsi_note = "Above 70" if last_rsi >= 70 else "Below 30" if last_rsi <= 30 else "Between 30 and 70"
                st.write(f"RSI range: **{rsi_note}**. RSI is a momentum indicator, not a buy/sell signal.")

    with bars_tab:
        recent = raw.tail(30).rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        }).rename_axis("Time (IST)").reset_index()
        st.dataframe(recent, hide_index=True, alt=f"Latest 30 five-minute bars for {name}")


show_live_analysis(symbol, instrument, visible_range)
