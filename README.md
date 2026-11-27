# AI-Powered Financial Risk and Market Prediction System

A machine learning mini project for exploring the NIFTY 50. The dashboard provides price and risk analytics, volatility forecasts, walk-forward backtests, five-minute market views, company/sector browsing, and ARIMA/SARIMA demonstrations. Stock prices are downloaded from Yahoo Finance.

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
├── stock_universe.py       # Shared NIFTY 50 list and NSE links
├── train.py
├── data/NIFTY50_demo_sample_5minute.csv
├── data/nifty50_stocks.csv # 50-stock snapshot with sector and browsing groups
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
1. Filter the NIFTY 50 list by sector or company group, then choose a company.
2. **Price & risk** tab: price chart, volatility, VaR, CVaR, max drawdown.
3. **Volatility forecast** tab: predicted risk for the next 5 days and which inputs the model uses.
4. **Backtest** tab: buy-and-hold vs the ML risk-aware strategy. Use the slider to change when the strategy moves to cash.

The first load of each stock takes 10 to 20 seconds (download and model training). After that it is cached.

## New analysis pages

The Streamlit sidebar includes **Dataset Analysis**, **Live Market**, **Market Chatbot**, and **ARIMA / SARIMA Forecast** pages. The main risk dashboard, Live Market, ARIMA/SARIMA page, and chatbot share a 50-company NIFTY 50 snapshot in `data/nifty50_stocks.csv`. Update the snapshot from the [official NSE NIFTY 50 page](https://www.nseindia.in/index-tracker/NIFTY%2050) when membership changes. Sector and company group labels are maintained for app filtering and are not official NSE classifications. Live Market also links to official NSE market movers, 52-week highs/lows, and company announcements.

Dataset Analysis uses the small five-session Nifty 50 sample CSV in `data/`; Live Market downloads five-minute quotes from Yahoo Finance and refreshes every five minutes. The sample is for demonstrating the data format, not for drawing statistically reliable conclusions.

### Market chatbot setup

The chatbot offers two providers: local Ollama and the Gemini API. It also attempts to fetch equity/index quotes from NSE's public endpoints. NSE may rate-limit or block automated requests; when that happens, the chatbot reports that it could not verify the quote and links to NSE's website. Quote timing and availability are controlled by the source.

Install [Ollama](https://ollama.com/download), then in PowerShell download and start the Qwen model:

```powershell
ollama run qwen3.5:4b
```

Keep Ollama running, open another terminal in the project, and start Streamlit with `python -m streamlit run finance-risk-app.py`. On the Market Chatbot page, select **Ollama (local)**, use **Check Ollama connection**, and make sure the model name is `qwen3.5:4b`. Ollama runs on the same computer as Streamlit; the Streamlit Community Cloud app cannot access Ollama running on your PC.

To use Gemini, create a key in [Google AI Studio](https://aistudio.google.com/api-keys). For local use, add `GEMINI_API_KEY = "your-key"` to `.streamlit/secrets.toml`. For Streamlit Community Cloud, add the same `GEMINI_API_KEY` setting under **App → Settings → Secrets**. Keep the key private and do not commit it. Select **Gemini API** on the chatbot page. Gemini free-tier quotas and model availability vary; check [current pricing and limits](https://ai.google.dev/gemini-api/docs/pricing/). Google indicates that free-tier prompts may be used to improve its products, so avoid sending private or sensitive information.

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
