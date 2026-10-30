from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_NAME = "Baldeh Scanner"
BENCHMARKS = {"SPY": "S&P 500"}
PERIOD_DAYS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

SECTORS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

THEMES = {
    "Semiconductors": "SOXX",
    "Cybersecurity": "CIBR",
    "Cloud Computing": "SKYY",
    "Software": "IGV",
    "AI & Robotics": "BOTZ",
    "Quantum Computing": "QTUM",
    "Biotechnology": "XBI",
    "Pharmaceuticals": "IHE",
    "Medical Devices": "IHI",
    "Healthcare Providers": "IHF",
    "Digital Payments": "IPAY",
    "Regional Banks": "KRE",
    "Insurance": "KIE",
    "E-commerce": "IBUY",
    "Autos & Electric Vehicles": "DRIV",
    "Travel & Leisure": "PEJ",
    "Oil & Gas E&P": "XOP",
    "Oil Services": "OIH",
    "Clean Energy": "ICLN",
    "Uranium & Nuclear": "URA",
    "Aerospace & Defense": "ITA",
    "Industrial Automation": "ROBO",
    "Transportation & Logistics": "IYT",
    "Copper Miners": "COPX",
    "Gold Miners": "GDX",
    "Lithium & Battery Materials": "LIT",
    "Data Centers & Digital Infrastructure": "SRVR",
    "REITs": "VNQ",
    "Solar Energy": "TAN",
    "Telecommunications": "IYZ",
    "Media & Entertainment": "PBS",
    "Social Media & Digital Advertising": "SOCL",
    "Video Games & Esports": "HERO",
}

STOCKS = {
    "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "AVGO": "Broadcom",
    "QCOM": "Qualcomm", "MSFT": "Microsoft", "ORCL": "Oracle", "CRM": "Salesforce",
    "NOW": "ServiceNow", "PLTR": "Palantir", "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike", "LLY": "Eli Lilly", "NVO": "Novo Nordisk",
    "ABBV": "AbbVie", "MRK": "Merck", "REGN": "Regeneron", "ISRG": "Intuitive Surgical",
    "UNH": "UnitedHealth", "JPM": "JPMorgan Chase", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal",
    "COIN": "Coinbase", "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot",
    "BKNG": "Booking Holdings", "ABNB": "Airbnb", "DASH": "DoorDash",
    "WMT": "Walmart", "COST": "Costco", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "SLB",
    "FSLR": "First Solar", "CEG": "Constellation Energy", "VST": "Vistra",
    "GE": "GE Aerospace", "RTX": "RTX", "LMT": "Lockheed Martin", "CAT": "Caterpillar",
    "ETN": "Eaton", "UPS": "UPS", "FCX": "Freeport-McMoRan", "NEM": "Newmont",
    "ALB": "Albemarle", "LIN": "Linde", "PLD": "Prologis", "EQIX": "Equinix",
    "DLR": "Digital Realty", "AMT": "American Tower", "NEE": "NextEra Energy",
    "DUK": "Duke Energy", "SO": "Southern Company", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "NFLX": "Netflix", "DIS": "Disney", "T": "AT&T",
    "VZ": "Verizon", "TMUS": "T-Mobile", "SPOT": "Spotify", "RBLX": "Roblox",
}


FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X", "USD/JPY": "JPY=X", "USD/CHF": "CHF=X",
    "USD/CAD": "CAD=X", "EUR/JPY": "EURJPY=X", "EUR/GBP": "EURGBP=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X",
    "GBP/CHF": "GBPCHF=X",
}

# Yahoo Finance futures symbols. Availability and intraday history can vary by contract.
COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cocoa": "CC=F",
    "Cotton": "CT=F",
}

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def forex_timeframe_signal(ticker, interval, period):
    candles = download_history(ticker, period, interval)
    if candles.empty or len(candles) < 55:
        return {"Signal": "INSUFFICIENT DATA", "Close": np.nan, "RSI": np.nan}
    close = candles["Close"].dropna()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    rsi = calculate_rsi(close)
    momentum = (close / close.shift(10) - 1) * 100
    price, e20, e50 = float(close.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
    mom_now = float(momentum.iloc[-1]) if pd.notna(momentum.iloc[-1]) else np.nan
    score = (1 if price > e20 else -1) + (1 if e20 > e50 else -1)
    score += 1 if np.isfinite(rsi_now) and rsi_now >= 50 else -1
    score += 1 if np.isfinite(mom_now) and mom_now > 0 else -1
    signal = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "MIXED"
    return {"Signal": signal, "Close": price, "RSI": rsi_now}

@st.cache_data(ttl=300, show_spinner=False)
def forex_scan_table():
    rows = []
    for pair, ticker in FOREX_PAIRS.items():
        hourly = forex_timeframe_signal(ticker, "60m", "60d")
        daily = forex_timeframe_signal(ticker, "1d", "2y")
        h, d = hourly["Signal"], daily["Signal"]
        if h == "BULLISH" and d == "BULLISH":
            classification = "BULLISH — BOTH TIMEFRAMES"
        elif h == "BEARISH" and d == "BEARISH":
            classification = "BEARISH — BOTH TIMEFRAMES"
        elif "INSUFFICIENT DATA" in (h, d):
            classification = "INSUFFICIENT DATA"
        else:
            classification = "MIXED — TIMEFRAMES DISAGREE"
        rows.append({"Currency Pair": pair, "Ticker": ticker, "1-Hour Signal": h,
                     "Daily Signal": d, "Relative Classification": classification,
                     "1H RSI": hourly["RSI"], "Daily RSI": daily["RSI"]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def commodity_scan_table():
    """Scan commodity futures using 1-hour and daily technical signals."""
    rows = []
    for name, ticker in COMMODITIES.items():
        hourly = forex_timeframe_signal(ticker, "60m", "60d")
        daily = forex_timeframe_signal(ticker, "1d", "2y")
        h, d = hourly["Signal"], daily["Signal"]
        if h == "BULLISH" and d == "BULLISH":
            classification = "BULLISH — BOTH TIMEFRAMES"
        elif h == "BEARISH" and d == "BEARISH":
            classification = "BEARISH — BOTH TIMEFRAMES"
        elif "INSUFFICIENT DATA" in (h, d):
            classification = "INSUFFICIENT DATA"
        else:
            classification = "MIXED — TIMEFRAMES DISAGREE"
        rows.append({"Commodity": name, "Ticker": ticker, "1-Hour Signal": h,
                     "Daily Signal": d, "Relative Classification": classification,
                     "1H RSI": hourly["RSI"], "Daily RSI": daily["RSI"]})
    return pd.DataFrame(rows)

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.signal-bull {color:#14804a;font-weight:700}
.signal-bear {color:#d62728;font-weight:700}
.signal-mixed {color:#b8860b;font-weight:700}
</style>
""", unsafe_allow_html=True)

def download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download one symbol; normalize yfinance's varying output shape."""
    try:
        raw = yf.download(
            ticker, period=period, interval=interval, auto_adjust=True,
            progress=False, threads=False, group_by="column"
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            # Single-symbol downloads can still return a MultiIndex in some versions.
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        out = raw[needed].copy()
        out = out.dropna(subset=["Close"])
        if out.index.tz is not None:
            out.index = out.index.tz_convert("America/New_York")
        return out
    except Exception:
        return pd.DataFrame()

def hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    root = max(1, int(np.sqrt(period)))
    weights_half = np.arange(1, half + 1, dtype=float)
    weights_full = np.arange(1, period + 1, dtype=float)
    weights_root = np.arange(1, root + 1, dtype=float)

    def wma(s: pd.Series, weights: np.ndarray) -> pd.Series:
        return s.rolling(len(weights), min_periods=len(weights)).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    return wma(2 * wma(series, weights_half) - wma(series, weights_full), weights_root)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["SMA 5"] = out["Close"].rolling(5, min_periods=5).mean()
    out["SMA 10"] = out["Close"].rolling(10, min_periods=10).mean()
    out["SMA 21"] = out["Close"].rolling(21, min_periods=21).mean()
    out["HMA 20"] = hma(out["Close"], 20)
    return out

def resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bars into 4-hour bars. Uses available hourly data and ET timestamps."""
    if hourly.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    cols = [c for c in agg if c in hourly.columns]
    # Anchor bars to the US market session; the first bar can be a partial session bar.
    result = hourly[cols].resample("4h", origin="start_day", offset="9h30min").agg(
        {k: v for k, v in agg.items() if k in cols}
    ).dropna(subset=["Close"])
    return result

def trend_snapshot(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 22:
        return {"Close": np.nan, "SMA 5": np.nan, "SMA 10": np.nan,
                "SMA 21": np.nan, "HMA 20": np.nan, "Score": 0,
                "Signal": "INSUFFICIENT DATA", "Conditions": 0}
    x = add_indicators(df).dropna(subset=["SMA 5", "SMA 10", "SMA 21", "HMA 20"])
    if x.empty:
        return {"Close": np.nan, "SMA 5": np.nan, "SMA 10": np.nan,
                "SMA 21": np.nan, "HMA 20": np.nan, "Score": 0,
                "Signal": "INSUFFICIENT DATA", "Conditions": 0}
    r = x.iloc[-1]
    price = float(r["Close"])
    tests = [
        price > r["SMA 5"],
        price > r["SMA 10"],
        price > r["SMA 21"],
        price > r["HMA 20"],
        r["SMA 5"] > r["SMA 10"],
        r["SMA 10"] > r["SMA 21"],
    ]
    score = sum(1 if t else -1 for t in tests)
    if score >= 4:
        signal = "BULLISH"
    elif score <= -4:
        signal = "BEARISH"
    else:
        signal = "MIXED"
    return {
        "Close": price, "SMA 5": float(r["SMA 5"]), "SMA 10": float(r["SMA 10"]),
        "SMA 21": float(r["SMA 21"]), "HMA 20": float(r["HMA 20"]),
        "Score": int(score), "Signal": signal, "Conditions": int(sum(tests)),
    }

def daily_return(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) <= days:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-days-1] - 1) * 100)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_close(ticker: str) -> pd.Series:
    df = download_history(ticker, "2y", "1d")
    return df["Close"].dropna() if not df.empty and "Close" in df else pd.Series(dtype=float)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, interval: str, period: str) -> pd.DataFrame:
    return download_history(ticker, period, interval)

@st.cache_data(ttl=900, show_spinner=False)
def relative_strength_table(kind: str) -> pd.DataFrame:
    if kind == "Sectors":
        universe = SECTORS
    elif kind == "Themes":
        universe = THEMES
    else:
        # STOCKS is stored as ticker -> company name; normalize for the ranking loop.
        universe = {symbol: symbol for symbol in STOCKS}
    symbols = sorted(set(universe.values()) | {"SPY"})
    closes = {s: fetch_daily_close(s) for s in symbols}
    rows = []
    for name, ticker in universe.items():
        display_name = STOCKS.get(name, name) if kind == "Stocks" else name
        s = closes.get(ticker, pd.Series(dtype=float))
        spy = closes.get("SPY", pd.Series(dtype=float))
        row = {"Name": display_name, "Ticker": ticker}
        for label, days in PERIOD_DAYS.items():
            asset_ret = daily_return(s, days)
            spy_ret = daily_return(spy, days)
            row[f"{label} Return %"] = asset_ret
            row[f"{label} vs SPY (pp)"] = asset_ret - spy_ret if np.isfinite(asset_ret) and np.isfinite(spy_ret) else np.nan
        vals = [row["3M vs SPY (pp)"], row["6M vs SPY (pp)"]]
        row["SPY RS Score"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        sp3, sp6 = row["3M vs SPY (pp)"], row["6M vs SPY (pp)"]
        if np.isfinite(sp3) and np.isfinite(sp6):
            row["Relative Classification"] = (
                "BULLISH" if sp3 > 0 and sp6 > 0 else
                "BEARISH" if sp3 < 0 and sp6 < 0 else "MIXED"
            )
        else:
            row["Relative Classification"] = "INSUFFICIENT DATA"
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("SPY RS Score", ascending=False, na_position="last").reset_index(drop=True)
        result.insert(0, "Rank", np.arange(1, len(result) + 1))
    return result

def fmt_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def signal_emoji(signal: str) -> str:
    return {"BULLISH": "🟢", "BEARISH": "🔴", "MIXED": "🟡"}.get(signal, "⚪")

def color_classification(df: pd.DataFrame):
    """Color classification cells green, red, or yellow for quick visual scanning."""
    if "Relative Classification" not in df.columns:
        return df
    def paint(value):
        value = str(value).upper()
        if "BULLISH" in value:
            return "background-color: #d9f2e3; color: #146c43; font-weight: 700"
        if "BEARISH" in value:
            return "background-color: #f8d7da; color: #842029; font-weight: 700"
        if "MIXED" in value:
            return "background-color: #fff3cd; color: #856404; font-weight: 700"
        return ""
    return df.style.map(paint, subset=["Relative Classification"])

st.title("📊 Baldeh Scanner")
st.caption("Multi-timeframe market intelligence • SPY benchmark only • No email alerts")

with st.sidebar:
    st.header("Scanner Controls")
    if st.button("🔄 Refresh market data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("**Benchmarks**")
    st.write("SPY — S&P 500")
    st.markdown("**Trend indicators**")
    st.write("5 SMA · 10 SMA · 21 SMA · 20 HMA")
    st.caption("Yahoo Finance data may be delayed or limited, especially for intraday candles.")

tabs = st.tabs(["Market Overview", "Sector Rankings", "Theme Rankings", "Stock Scanner", "Forex Scanner", "Commodity Scanner", "Methodology"])

with tabs[0]:
    st.subheader("SPY: Multi-timeframe trend")
    st.write("The 1-hour and daily candles are downloaded separately. Four-hour candles are aggregated from available hourly bars.")
    selected = "SPY"
    with st.spinner(f"Loading {selected} market data..."):
        hourly = fetch_intraday(selected, "60m", "60d")
        daily = fetch_intraday(selected, "1d", "2y")
        four_hour = resample_4h(hourly) if not hourly.empty else pd.DataFrame()
        snapshots = {
            "1 Hour": trend_snapshot(hourly),
            "4 Hours": trend_snapshot(four_hour),
            "Daily": trend_snapshot(daily),
        }
    c1, c2, c3 = st.columns(3)
    for col, (tf, snap) in zip([c1, c2, c3], snapshots.items()):
        with col:
            st.metric(tf, f"{signal_emoji(snap['Signal'])} {snap['Signal']}", f"Score {snap['Score']:+d}/6")
            st.write(f"Last close: {fmt_money(snap['Close'])}")
            details = pd.DataFrame({
                "Indicator": ["5 SMA", "10 SMA", "21 SMA", "20 HMA"],
                "Value": [snap["SMA 5"], snap["SMA 10"], snap["SMA 21"], snap["HMA 20"]],
            })
            details["Value"] = details["Value"].map(fmt_money)
            st.dataframe(details, hide_index=True, use_container_width=True)
    scores = [v["Score"] for v in snapshots.values() if v["Signal"] != "INSUFFICIENT DATA"]
    if len(scores) == 3:
        combined = sum(scores)
        overall = "BULLISH" if combined >= 7 else "BEARISH" if combined <= -7 else "MIXED"
        st.metric(f"{selected} Combined Multi-Timeframe Signal", f"{signal_emoji(overall)} {overall}", f"Combined score {combined:+d}/18")
    else:
        st.warning("Some timeframes do not have enough data to calculate all four indicators. No combined signal is shown.")
    st.caption(f"Data checked: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Signals use the latest candle returned by the data provider; the current intraday candle may still be forming.")

with tabs[1]:
    st.subheader("US sector relative strength")
    with st.spinner("Calculating sector rankings..."):
        df = relative_strength_table("Sectors")
    if df.empty:
        st.error("No sector data returned. Try refreshing later.")
    else:
        st.caption("Relative-strength values are percentage-point differences versus SPY. Positive values mean outperformance.")
        st.dataframe(color_classification(df), use_container_width=True, hide_index=True)
        st.download_button("Download sector rankings CSV", df.to_csv(index=False), "baldeh_sector_rankings.csv", "text/csv")

with tabs[2]:
    st.subheader("Investment theme relative strength")
    with st.spinner("Calculating theme rankings..."):
        df = relative_strength_table("Themes")
    if df.empty:
        st.error("No theme data returned. Try refreshing later.")
    else:
        st.caption("Ranked by the average 3-month and 6-month relative performance versus SPY.")
        st.dataframe(color_classification(df), use_container_width=True, hide_index=True)
        st.download_button("Download theme rankings CSV", df.to_csv(index=False), "baldeh_theme_rankings.csv", "text/csv")

with tabs[3]:
    st.subheader("Individual stock scanner")
    search = st.text_input("Filter by ticker or company", placeholder="e.g. NVDA or NVIDIA")
    with st.spinner("Calculating stock rankings..."):
        df = relative_strength_table("Stocks")
    if not df.empty:
        if search.strip():
            mask = df["Ticker"].str.contains(search.strip(), case=False, na=False) | df["Name"].str.contains(search.strip(), case=False, na=False)
            df = df[mask]
        st.caption("The stock universe is curated, not exhaustive. Rankings are research signals, not buy/sell instructions.")
        stock_view = df[["Name", "Ticker", "Relative Classification"]].rename(columns={"Name": "Stock Name"})
        st.dataframe(color_classification(stock_view), use_container_width=True, hide_index=True)
        st.download_button("Download stock classifications CSV", stock_view.to_csv(index=False), "baldeh_stock_classifications.csv", "text/csv")
    else:
        st.error("No stock data returned. Try refreshing later.")

with tabs[4]:
    st.subheader("Forex Scanner")
    st.caption("Major forex pairs and selected crosses. Trend uses 1-hour and daily candles, EMA 20/50, RSI(14), and 10-candle momentum.")
    if st.button("Refresh forex data", key="refresh_forex"):
        st.cache_data.clear()
        st.rerun()
    with st.spinner("Scanning forex pairs..."):
        fx = forex_scan_table()
    if fx.empty:
        st.error("No forex data returned. Try refreshing later.")
    else:
        st.dataframe(color_classification(fx), use_container_width=True, hide_index=True)
        st.download_button("Download forex scan CSV", fx.to_csv(index=False), "baldeh_forex_scan.csv", "text/csv")

with tabs[5]:
    st.subheader("Commodity Scanner")
    st.caption("Precious metals, energy, industrial metals, and agricultural futures. Signals use 1-hour and daily candles, EMA 20/50, RSI(14), and 10-candle momentum.")
    if st.button("Refresh commodity data", key="refresh_commodities"):
        st.cache_data.clear()
        st.rerun()
    with st.spinner("Scanning commodity futures..."):
        commodities = commodity_scan_table()
    if commodities.empty:
        st.error("No commodity data returned. Try refreshing later.")
    else:
        st.dataframe(color_classification(commodities), use_container_width=True, hide_index=True)
        st.download_button("Download commodity scan CSV", commodities.to_csv(index=False), "baldeh_commodity_scan.csv", "text/csv")

with tabs[6]:
    st.subheader("Methodology and limitations")
    st.markdown("""
**Benchmark**
- **SPY:** S&P 500 ETF benchmark. All stock, sector, and theme comparisons use SPY only.

**Relative strength**
- Each sector, theme ETF, or stock's total return is measured over 1, 3, 6, and 12 months.
- Relative strength versus a benchmark is asset return minus benchmark return, expressed in percentage points.
- The **SPY RS Score** averages the 3-month and 6-month relative returns versus SPY. It is a ranking metric, not a forecast.
- **BULLISH** means both 3-month and 6-month relative returns beat SPY; **BEARISH** means both underperform SPY; **MIXED** means the periods disagree or are not both positive/negative.

**Trend indicators**
- Market benchmarks use 5 SMA, 10 SMA, 21 SMA, and 20 HMA on 1-hour, 4-hour, and daily candles.
- Four-hour bars are aggregated from hourly bars; session boundaries and incomplete candles can affect intraday readings.

**Forex**
- Forex pairs use 1-hour and daily candles with EMA 20, EMA 50, RSI(14), and 10-candle momentum.
- A bullish or bearish classification requires both timeframes to agree; disagreement is shown as mixed.
- Forex data availability and intraday history are controlled by Yahoo Finance.

**Commodities**
- Commodity futures use 1-hour and daily candles with EMA 20, EMA 50, RSI(14), and 10-candle momentum.
- A bullish or bearish classification requires both timeframes to agree; disagreement is shown as mixed. Futures symbols may have limited or unavailable intraday data, and continuous futures prices are not the same as a specific tradable contract.

**Data**
- Prices are obtained from Yahoo Finance through `yfinance`; availability, timestamps, and historical coverage are controlled by the provider.
- The app caches results briefly to reduce repeated requests. Use **Refresh market data** to clear the cache and request new values.
- If data is missing or stale, verify the provider's response before relying on the reading.

**Important**
This tool is for market research and education, not financial advice. Signals are rule-based and may be wrong. Always independently verify data and manage risk.
""")
    st.write("No email-alert code, SMTP settings, or scheduled email workflow is included in this project.")
