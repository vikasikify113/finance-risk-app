"""Educational ARIMA and SARIMA forecasts for daily NSE stock prices."""

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ARIMA / SARIMA Forecast", page_icon="📈", layout="wide")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    st.error("statsmodels is required. Install the project dependencies with `pip install -r requirements.txt`.")
    st.stop()

st.title("ARIMA and SARIMA Price Forecast")
st.caption(
    "Forecasts are statistical estimates from historical daily prices. They are not reliable "
    "promises of future prices or investment advice."
)

NSE_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "WIPRO": "WIPRO.NS",
    "LT": "LT.NS",
    "MARUTI": "MARUTI.NS",
}


@st.cache_data(ttl=900, show_spinner="Downloading daily prices from Yahoo Finance…")
def load_daily_prices(yahoo_ticker: str) -> pd.Series:
    frame = yf.download(
        yahoo_ticker, period="5y", interval="1d", auto_adjust=True,
        progress=False, threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"No daily closing-price data was returned for {yahoo_ticker}.")

    close_column = next(
        (
            column for column in frame.columns
            if "close" in [str(part).lower() for part in (column if isinstance(column, tuple) else (column,))]
        ),
        None,
    )
    if close_column is None:
        raise ValueError(f"No daily closing-price data was returned for {yahoo_ticker}.")
    close = frame[close_column]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close[~close.index.duplicated(keep="last")].sort_index()


with st.sidebar:
    selected_symbol = st.selectbox("NSE symbol", list(NSE_TICKERS))
    custom_symbol = st.text_input("Or enter another Yahoo Finance symbol", placeholder="e.g. ADANIENT.NS")
    yahoo_ticker = custom_symbol.strip().upper() if custom_symbol.strip() else NSE_TICKERS[selected_symbol]
    horizon = st.slider("Forecast trading days", min_value=1, max_value=30, value=10)
    model_choice = st.radio("Model", ["ARIMA", "SARIMA"], horizontal=True)
    order_p = st.select_slider("ARIMA p", options=[0, 1, 2, 3], value=1)
    order_d = st.select_slider("ARIMA d", options=[0, 1, 2], value=1)
    order_q = st.select_slider("ARIMA q", options=[0, 1, 2, 3], value=1)
    if model_choice == "SARIMA":
        st.caption("Seasonality defaults to a 5-session trading week.")

try:
    close = load_daily_prices(yahoo_ticker)
except Exception as exc:
    st.error(f"Could not load daily prices: {exc}")
    st.stop()

if len(close) < 100:
    st.error("At least 100 daily prices are needed for this demonstration. Choose another symbol.")
    st.stop()

st.subheader(f"{yahoo_ticker}: daily adjusted closing price")
st.line_chart(close.rename("Adjusted close"))
st.write(f"Using {len(close):,} daily observations from {close.index.min():%d %b %Y} to {close.index.max():%d %b %Y}.")

if st.button("Fit model and forecast", type="primary"):
    order = (int(order_p), int(order_d), int(order_q))
    with st.spinner(f"Fitting {model_choice} model…"):
        try:
            if model_choice == "ARIMA":
                fitted = ARIMA(close, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
            else:
                fitted = SARIMAX(
                    close,
                    order=order,
                    seasonal_order=(1, 1, 1, 5),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=100)
            forecast = fitted.get_forecast(steps=horizon)
            mean = forecast.predicted_mean
            interval = forecast.conf_int(alpha=0.05)
            future_dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
            forecast_frame = pd.DataFrame(
                {
                    "Forecast": mean.to_numpy(),
                    "Lower 95% interval": interval.iloc[:, 0].to_numpy(),
                    "Upper 95% interval": interval.iloc[:, 1].to_numpy(),
                },
                index=future_dates,
            )
            chart = pd.concat(
                [close.tail(120).rename("Historical close"), forecast_frame], axis=1
            )
            st.subheader(f"{model_choice} {order} forecast")
            st.line_chart(chart, y=["Historical close", "Forecast", "Lower 95% interval", "Upper 95% interval"])
            st.dataframe(forecast_frame.style.format("₹{:,.2f}"), hide_index=False)
            st.caption(
                "SARIMA uses a 5-session seasonal period in this demo. The shaded-style interval "
                "columns show a model-based 95% interval; real outcomes may fall outside it. "
                "This simple model does not include news, fundamentals, transaction costs, or market shocks."
            )
        except Exception as exc:
            st.error(f"Model fitting failed for this symbol and order: {exc}")

