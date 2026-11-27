"""Market Q&A chatbot using local Ollama and best-effort NSE quote data."""

import os
import re
from datetime import datetime
from urllib.parse import quote

import requests
import streamlit as st
from google import genai
from openai import OpenAI
from stock_universe import NSE_LIVE_MARKET, load_nifty50

st.set_page_config(page_title="Market Chatbot", page_icon="💬")
st.title("Market Chatbot")
st.caption(
    "Ask about an NSE-listed stock or the Indian market. Choose local Ollama or the Gemini API "
    "in the sidebar. Quote data may be delayed or temporarily unavailable. "
    "Educational use only; not investment advice."
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_BASE_URL = f"{OLLAMA_HOST}/v1"
GEMINI_MODEL = "gemini-3.8-flash"

NSE_HOME = NSE_LIVE_MARKET
NSE_QUOTE = "https://www.nseindia.in/get-quotes/equity?symbol={}"
STOCKS = load_nifty50()
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
    "DOING", "PERFORMANCE", "PERFORM", "LOOKING", "THANKS", "THANK", "HI", "HEY", "HELLO",
    "COMPANY", "COMPANIES", "SECTOR", "BANKING", "STOCKS", "GOVERNMENT",
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
    # Match the checked-in NIFTY 50 snapshot by ticker or company name.
    matched = []
    for row in STOCKS.itertuples(index=False):
        if any(
            re.search(rf"(?<![A-Z0-9]){re.escape(value)}(?![A-Z0-9])", text, re.IGNORECASE)
            for value in (row.symbol, row.company)
        ):
            matched.append(row.symbol)
    if matched:
        return matched[:3]

    # Keep support for user-entered tickers outside this NIFTY 50 snapshot.
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9&-]{1,14}\b", text)
    return list(dict.fromkeys(word.upper() for word in words if word.upper() not in STOP_WORDS))[:3]


def nse_quote_url(symbol: str) -> str:
    return NSE_QUOTE.format(quote(symbol, safe=""))


def format_price(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


if "market_chat" not in st.session_state:
    st.session_state.market_chat = []

provider = st.sidebar.radio("Chat provider", ["Ollama (local)", "Gemini API"])
previous_provider = st.session_state.get("last_chat_provider")
if previous_provider and previous_provider != provider:
    st.session_state.market_chat = []
    st.session_state.pop("gemini_previous_interaction_id", None)
st.session_state.last_chat_provider = provider

if provider == "Ollama (local)":
    model = st.sidebar.text_input(
        "Ollama model name", value="qwen3.5:4b", help="Use a model shown by `ollama list`."
    )
    st.sidebar.caption(f"Local Ollama endpoint: {OLLAMA_HOST}")
    if st.sidebar.button("Check Ollama connection"):
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            response.raise_for_status()
            installed_models = [item.get("name", "") for item in response.json().get("models", [])]
            st.sidebar.success("Ollama is running.")
            if model in installed_models or any(
                item.split(":")[0] == model.split(":")[0] for item in installed_models
            ):
                st.sidebar.success(f"Model `{model}` is installed.")
            else:
                st.sidebar.warning(
                    f"Model `{model}` was not found. Installed: {', '.join(installed_models) or 'none'}"
                )
        except Exception as exc:
            st.sidebar.error(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}")
            st.sidebar.caption("Start Ollama on this computer and run the Streamlit app locally.")
else:
    model = GEMINI_MODEL
    st.sidebar.caption(f"Gemini model: `{GEMINI_MODEL}`")
    st.sidebar.caption("Gemini requests send your prompt and relevant market context to Google.")
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.sidebar.warning("Add GEMINI_API_KEY in Streamlit Secrets to use Gemini.")
st.sidebar.link_button("Open NSE live equity market", NSE_HOME)
st.sidebar.divider()
quote_symbol_input = st.sidebar.text_input("Look up any NSE equity symbol", placeholder="e.g. RELIANCE")
if st.sidebar.button("Fetch NSE quote") and quote_symbol_input.strip():
    st.session_state.nse_quote_symbol = quote_symbol_input.strip().upper()
if st.sidebar.button("Clear conversation"):
    st.session_state.market_chat = []
    st.session_state.pop("gemini_previous_interaction_id", None)

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

for message in st.session_state.market_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about NIFTY, RELIANCE, TCS, or another NSE symbol…")
if prompt:
    st.session_state.market_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        symbols = symbols_in(prompt)
        upper_prompt = prompt.upper()
        asks_for_indices = any(
            term in upper_prompt for term in ("NIFTY", "INDEX", "INDICES", "VIX", "MARKET")
        )
        with st.spinner("Checking NSE quote data…"):
            quotes = [get_quote(symbol) for symbol in symbols]
            indices = get_indices() if asks_for_indices else []
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
            if provider == "Gemini API":
                if not gemini_api_key:
                    raise ValueError("GEMINI_API_KEY is missing from Streamlit Secrets or the environment.")
                gemini_client = genai.Client(api_key=gemini_api_key)
                request_args = {
                    "model": GEMINI_MODEL,
                    "system_instruction": instructions,
                    "input": (
                        f"User question: {prompt}\n\n"
                        f"Market data context (use only when relevant): {data_context}"
                    ),
                }
                previous_id = st.session_state.get("gemini_previous_interaction_id")
                if previous_id:
                    request_args["previous_interaction_id"] = previous_id
                with st.spinner(f"Waiting for Gemini model `{GEMINI_MODEL}`…"):
                    response = gemini_client.interactions.create(**request_args)
                reply = response.output_text or "Gemini returned an empty response. Please try again."
                st.session_state.gemini_previous_interaction_id = response.id
            else:
                ollama_client = OpenAI(
                    api_key="ollama",  # Ollama's OpenAI-compatible local API does not use a key.
                    base_url=OLLAMA_BASE_URL,
                    timeout=300,
                    max_retries=0,
                )
                with st.spinner(f"Waiting for local model `{model.strip()}`…"):
                    response = ollama_client.chat.completions.create(
                        model=model.strip(),
                        messages=[{"role": "system", "content": instructions}, *history],
                    )
                reply = response.choices[0].message.content or "I could not generate a response. Please try again."
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
            st.error(f"{provider} request failed: {detail or exc}")
            if provider == "Ollama (local)":
                st.caption(
                    "Check Ollama connection, confirm the model name with `ollama list`, "
                    "and keep this dashboard running on the same computer as Ollama."
                )
            else:
                st.caption("Check that GEMINI_API_KEY is valid and that the selected API model is enabled for your account.")

