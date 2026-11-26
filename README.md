# AI-Powered Financial Risk and Market Prediction System

A machine learning mini project on 15 NSE-listed stocks (2020 to 2026). It tests whether stock direction can be predicted, forecasts short-term volatility, groups stocks by risk, and backtests a risk-aware strategy. Everything is explorable in a Streamlit dashboard.

> Educational project. Not financial advice.

## What the project does
- **Direction prediction:** Logistic Regression, Random Forest and Gradient Boosting with walk-forward validation. All scored about 50%, no better than a baseline.
- **Volatility forecasting:** Ridge and Random Forest predict the next 5 days' volatility, with about 5% lower error than a naive baseline.
- **Risk analytics:** VaR, CVaR, max drawdown, Sharpe ratio, and K-Means risk clusters.
- **Backtest:** a risk-aware strategy vs buy-and-hold, out-of-sample, with transaction costs.
- **Dashboard:** price and risk metrics, volatility forecast, and backtest for each stock.

## Project structure
```text
finance-risk-app/
├── finance-risk-app.py  # Existing Streamlit dashboard
├── pages/
│   ├── 1_Dataset_Analysis.py
│   ├── 2_Live_Market.py
│   ├── 3_Market_Chatbot.py
│   └── 4_ARIMA_Forecast.py
├── features.py
├── train.py
├── data/NIFTY50_demo_sample_5minute.csv
├── requirements.txt
└── README.md
```
**Live demo:** https://your-app-name.streamlit.app
## Run it on your computer (localhost)

**Requirements:** Python 3.9 or newer, Git, and an internet connection (stock prices are downloaded live from Yahoo Finance).

**1. Clone the repository**
```
git clone https://github.com/vikasikify113/finance-risk-app.git
cd finance-risk-app
```

**2. (Recommended) Create a virtual environment**

Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the dependencies**
```
python -m pip install -r requirements.txt
```

**4. Start the dashboard**
```
python -m streamlit run finance-risk-app.py
```

**5. Open the app.** Your browser opens automatically. If it doesn't, go to http://localhost:8501.

To stop the app, press `Ctrl+C` in the terminal.

## Using the dashboard
1. Pick a stock in the sidebar.
2. **Price & risk** tab: price chart, volatility, VaR, CVaR, max drawdown.
3. **Volatility forecast** tab: predicted risk for the next 5 days and which inputs the model uses.
4. **Backtest** tab: buy-and-hold vs the ML risk-aware strategy. Use the slider to change when the strategy moves to cash.

The first load of each stock takes 10 to 20 seconds (download and model training). After that it is cached.

## New analysis pages

The Streamlit sidebar includes **Dataset Analysis**, **Live Market**, **Market Chatbot**, and **ARIMA / SARIMA Forecast** pages. Dataset Analysis uses the small five-session Nifty 50 sample CSV in `data/`; Live Market downloads five-minute quotes from Yahoo Finance and refreshes every five minutes. Both market pages include a link to NSE India. The sample is for demonstrating the data format, not for drawing statistically reliable conclusions.

### Market chatbot setup

The chatbot uses the xAI Grok API through its OpenAI-compatible Python client and attempts to fetch current equity/index quotes from NSE's public endpoints. The xAI API supports the OpenAI-compatible interface and recommends the Responses API for new integrations. NSE may rate-limit or block automated requests; when that happens, the chatbot reports that it could not verify the quote and links to NSE's website. Quote timing and availability are controlled by the source.

For local development, create `.streamlit/secrets.toml` and add:

```toml
XAI_API_KEY = "your-xai-api-key"
```

Keep this file private and do not commit it. On Streamlit Community Cloud, add `XAI_API_KEY` in the app's **Settings → Secrets**. xAI API usage may incur charges. Without a configured key, the chatbot page loads but Grok replies are disabled.

### ARIMA / SARIMA forecast

The forecast page downloads daily adjusted prices from Yahoo Finance. Choose one of the listed NSE stocks or enter another Yahoo Finance symbol such as `ADANIENT.NS`. ARIMA uses the selected `(p, d, q)` order; SARIMA also uses a five-session seasonal pattern for this demonstration. Forecasts are estimates based only on past prices, and are not investment advice.

To train the optional intraday model locally, add the full historical `NIFTYMS400_MQ100_5minute.csv` to `data/` and run `python train.py`. This creates files in `models/`; the dataset page shows model results when those files are available. The included five-session sample is not intended for training.

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` is not recognized | Use `python -m streamlit run finance-risk-app.py` |
| `python` is not recognized (Windows) | Use `py` instead, for example `py -m streamlit run finance-risk-app.py` |
| "File does not exist: finance-risk-app.py" | Run the command from inside the `finance-risk-app` folder |
| Port 8501 already in use | `python -m streamlit run finance-risk-app.py --server.port 8502` |
| Empty chart or "Could not download data" | Check your internet connection and try again, Yahoo Finance sometimes rate-limits |

## Notebook analysis
The notebook mentioned in older documentation is not included in the current repository.

## Tech stack
Python, pandas, NumPy, scikit-learn, yfinance, Matplotlib, Streamlit, Google Colab
