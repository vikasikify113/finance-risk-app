"""Shared NIFTY 50 snapshot and NSE links used by dashboard pages."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
STOCKS_CSV = PROJECT_ROOT / "data" / "nifty50_stocks.csv"
NSE_LIVE_MARKET = "https://www.nseindia.in/market-data/live-equity-market"
NSE_NIFTY50 = "https://www.nseindia.in/index-tracker/NIFTY%2050"
NSE_TOP_MOVERS = "https://www.nseindia.in/market-data/top-gainers-losers"
NSE_52_WEEK = "https://www.nseindia.in/market-data/52-week-high-equity-market"
NSE_CORPORATE_FILINGS = "https://www.nseindia.in/companies-listing/corporate-filings-announcements"


def load_nifty50() -> pd.DataFrame:
    """Load the checked-in NIFTY 50 snapshot with stable column types."""
    stocks = pd.read_csv(STOCKS_CSV).fillna("")
    required = {"symbol", "company", "sector", "company_group", "yahoo_ticker"}
    missing = required.difference(stocks.columns)
    if missing:
        raise ValueError(f"Stock list is missing columns: {', '.join(sorted(missing))}")
    return stocks
