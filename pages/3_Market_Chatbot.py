"""Market Q&A chatbot using OpenAI and best-effort NSE quote data."""

import os
import re
from datetime import datetime
from urllib.parse import quote

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Market Chatbot", page_icon="💬")
st.title("Market Chatbot")
st.caption(
    "Ask about an NSE-listed stock or the Indian market. Quote data may be delayed "
    "or temporarily unavailable. This tool is for education, not investment advice."
)

NSE_HOME = "https://www.nseindia.com/market-data/live-equity-market"
NSE_QUOTE = "https://www.nseindia.com/get-quotes/equity?symbol={}"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
STOP_WORDS = {
    "NSE", "BSE", "NIFTY", "BANK", "INDIA", "VIX", "PRICE", "STOCK",
    "SHARE", "MARKET", "TODAY", "CURRENT", "WHAT", "HOW", "ABOUT", "IS",
    "THE", "AND", "FOR", "OF", "TO", "IN", "ON", "CAN", "YOU", "PLEASE",
    "TELL", "ME", "SHOW", "COMPARE", "WITH", "NEWS", "RISK", "ANALYSIS",
    "DOING", "PERFORMANCE", "PERFORM", "LOOKING", "TODAY", "PLEASE", "THANKS",
}


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(symbol: str) -> dict:
    """Fetch a best-effort quote from NSE's public quote endpoint."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        response = session.get(
            "https://www.nseindia.com/api/quote-equity",
            params={"symbol": symbol},
            timeout=10,
        )
        response.raise_for_status()
        price = response.json()["priceInfo"]
        return {
            "symbol": symbol,
            "last_price": price.get("lastPrice"),
            "change": price.get("change"),
            "pct_change": price.get("pChange"),
            "day_high": price.get("intraDayHighLow", {}).get("max"),
            "day_low": price.get("intraDayHighLow", {}).get("min"),
            "prev_close": price.get("previousClose"),
            "52w_high": price.get("weekHighLow", {}).get("max"),
            "52w_low": price.get("weekHighLow", {}).get("min"),
            "source": "NSE India quote endpoint",
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return {"symbol": symbol, "error": str(exc)}
    finally:
        session.close()


@st.cache_data(ttl=60, show_spinner=False)
def get_indices() -> list[dict]:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        response = session.get("https://www.nseindia.com/api/allIndices", timeout=10)
        response.raise_for_status()
        keep = {"NIFTY 50", "NIFTY BANK", "NIFTY IT", "INDIA VIX"}
        return [
            {"index": item["index"], "last": item["last"], "pct_change": item["percentChange"]}
            for item in response.json().get("data", [])
            if item.get("index") in keep
        ]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return [{"error": str(exc)}]
    finally:
        session.close()


def symbols_in(text: str) -> list[str]:
    # Uppercase the words so both “reliance” and “RELIANCE” are accepted.
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9&-]{1,14}\b", text)
    return list(dict.fromkeys(word.upper() for word in words if word.upper() not in STOP_WORDS))[:3]


def nse_quote_url(symbol: str) -> str:
    return NSE_QUOTE.format(quote(symbol, safe=""))


def format_price(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


api_key = os.getenv("XAI_API_KEY")
try:
    api_key = api_key or st.secrets.get("XAI_API_KEY")
except Exception:
    # No local secrets.toml is a normal setup; the chatbot gives setup guidance below.
    pass
model = "grok-4.7"
st.sidebar.caption("Add XAI_API_KEY to Streamlit secrets or your environment.")
st.sidebar.link_button("Open NSE live equity market", NSE_HOME)
st.sidebar.divider()
quote_symbol_input = st.sidebar.text_input("Look up any NSE equity symbol", placeholder="e.g. RELIANCE")
if st.sidebar.button("Fetch NSE quote") and quote_symbol_input.strip():
    st.session_state.nse_quote_symbol = quote_symbol_input.strip().upper()
if st.sidebar.button("Clear conversation"):
    st.session_state.pop("market_chat", None)

if st.session_state.get("nse_quote_symbol"):
    quote_symbol = st.session_state.nse_quote_symbol
    quote_data = get_quote(quote_symbol)
    st.subheader(f"NSE quote: {quote_symbol}")
    if quote_data.get("error"):
        st.warning("NSE could not return this quote through the app. Use the official NSE quote page below.")
    else:
        price_cols = st.columns(4)
        try:
            pct_delta = f"{float(quote_data['pct_change']):+.2f}%"
        except (TypeError, ValueError):
            pct_delta = None
        price_cols[0].metric("Last price", f"₹{format_price(quote_data['last_price'])}")
        price_cols[1].metric("Change", f"₹{format_price(quote_data['change'])}", pct_delta)
        price_cols[2].metric("Day high / low", f"{format_price(quote_data['day_high'])} / {format_price(quote_data['day_low'])}")
        price_cols[3].metric("52-week high / low", f"{format_price(quote_data['52w_high'])} / {format_price(quote_data['52w_low'])}")
    st.link_button(f"View {quote_symbol} on NSE India", nse_quote_url(quote_symbol))

if "market_chat" not in st.session_state:
    st.session_state.market_chat = []

for message in st.session_state.market_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about NIFTY, RELIANCE, TCS, or another NSE symbol…")
if prompt:
    st.session_state.market_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error(
                "Grok API key is not configured. Set XAI_API_KEY in the app's "
                "Streamlit secrets before using GPT."
            )
        else:
            symbols = symbols_in(prompt)
            with st.spinner("Checking NSE quote data…"):
                quotes = [get_quote(symbol) for symbol in symbols]
                indices = get_indices() if not symbols or any(
                    word in prompt.upper() for word in ("NIFTY", "INDEX", "INDICES", "VIX")
                ) else []
            data_context = {
                "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "NSE_quotes": quotes,
                "NSE_indices": indices,
            }
            instructions = (
                "You are an educational assistant for Indian stock-market questions. "
                "Use only the supplied market data for current prices. If a quote has an error "
                "or no quote was fetched, say current price could not be verified and point the "
                "user to the official NSE link. Never invent a price, guarantee a forecast, or "
                "give personalized buy/sell advice. Explain that quotes can be delayed. "
                f"Market data retrieved at {data_context['retrieved_at']}: {data_context}"
            )
            history = [
                {"role": item["role"], "content": item["content"]}
                for item in st.session_state.market_chat[-10:]
            ]
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1",
                    timeout=60,
                    max_retries=2,
                )
                # xAI's Responses API accepts system instructions as a system
                # message in `input` (rather than the OpenAI `instructions` field).
                response = client.responses.create(
                    model=model,
                    input=[{"role": "system", "content": instructions}, *history],
                    store=False,
                )
                reply = response.output_text or "I could not generate a response. Please try again."
                st.markdown(reply)
                for symbol in symbols:
                    st.link_button(f"View {symbol} on NSE India", nse_quote_url(symbol))
                st.session_state.market_chat.append({"role": "assistant", "content": reply})
            except Exception as exc:
                detail = getattr(exc, "body", None) or getattr(exc, "message", None)
                error_response = getattr(exc, "response", None)
                if error_response is not None:
                    try:
                        detail = error_response.json()
                    except Exception:
                        detail = getattr(error_response, "text", None) or detail
                st.error(f"The Grok request failed: {detail or exc}")

