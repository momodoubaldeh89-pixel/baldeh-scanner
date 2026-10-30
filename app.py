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

def technical_signal_from_candles(candles: pd.DataFrame) -> dict:
    """Return a rule-based trend signal and supporting values from candles."""
    if candles.empty or "Close" not in candles or candles["Close"].dropna().shape[0] < 55:
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
    return {"Signal": signal, "Close": price, "RSI": rsi_now, "Score": score}

def forex_timeframe_signal(ticker, interval, period):
    return technical_signal_from_candles(download_history(ticker, period, interval))

def multi_timeframe_classification(signals):
    valid = [x for x in signals if x != "INSUFFICIENT DATA"]
    if len(valid) < 2:
        return "INSUFFICIENT DATA"
    bulls, bears = valid.count("BULLISH"), valid.count("BEARISH")
    if bulls >= 2 and bulls > bears and len(valid) == 3:
        return "BULLISH — MULTI-TIMEFRAME"
    if bears >= 2 and bears > bulls and len(valid) == 3:
        return "BEARISH — MULTI-TIMEFRAME"
    if len(valid) == 2 and bulls == 2:
        return "BULLISH — BOTH TIMEFRAMES"
    if len(valid) == 2 and bears == 2:
        return "BEARISH — BOTH TIMEFRAMES"
    if bulls == len(valid):
        return "BULLISH — AVAILABLE TIMEFRAMES"
    if bears == len(valid):
        return "BEARISH — AVAILABLE TIMEFRAMES"
    return "MIXED — TIMEFRAMES DISAGREE"

def alignment_quality(signals):
    valid = [x for x in signals if x != "INSUFFICIENT DATA"]
    if not valid:
        return np.nan
    # Alignment score, not a probability of profit.
    return int(round(100 * max(valid.count("BULLISH"), valid.count("BEARISH")) / len(valid)))

@st.cache_data(ttl=300, show_spinner=False)
def forex_scan_table():
    rows = []
    for pair, ticker in FOREX_PAIRS.items():
        hourly_df = download_history(ticker, "60d", "60m")
        daily_df = download_history(ticker, "2y", "1d")
        four_df = resample_4h(hourly_df) if not hourly_df.empty else pd.DataFrame()
        h, f, d = (technical_signal_from_candles(x) for x in (hourly_df, four_df, daily_df))
        signals = [h["Signal"], f["Signal"], d["Signal"]]
        rows.append({"Currency Pair": pair, "Ticker": ticker, "1-Hour Signal": signals[0],
                     "4-Hour Signal": signals[1], "Daily Signal": signals[2],
                     "Relative Classification": multi_timeframe_classification(signals),
                     "Signal Quality / Alignment": alignment_quality(signals),
                     "Last Price": d["Close"], "1H RSI": h["RSI"], "Daily RSI": d["RSI"],
                     "Data Status": "OK" if sum(x != "INSUFFICIENT DATA" for x in signals) >= 2 else "INSUFFICIENT DATA"})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def commodity_scan_table():
    """Scan commodity futures using 1-hour, aggregated 4-hour, and daily signals."""
    rows = []
    for name, ticker in COMMODITIES.items():
        hourly_df = download_history(ticker, "60d", "60m")
        daily_df = download_history(ticker, "2y", "1d")
        four_df = resample_4h(hourly_df) if not hourly_df.empty else pd.DataFrame()
        h, f, d = (technical_signal_from_candles(x) for x in (hourly_df, four_df, daily_df))
        signals = [h["Signal"], f["Signal"], d["Signal"]]
        rows.append({"Commodity": name, "Ticker": ticker, "1-Hour Signal": signals[0],
                     "4-Hour Signal": signals[1], "Daily Signal": signals[2],
                     "Relative Classification": multi_timeframe_classification(signals),
                     "Signal Quality / Alignment": alignment_quality(signals),
                     "Last Price": d["Close"], "1H RSI": h["RSI"], "Daily RSI": d["RSI"],
                     "Data Status": "OK" if sum(x != "INSUFFICIENT DATA" for x in signals) >= 2 else "INSUFFICIENT DATA"})
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

def stock_quality_score(close: pd.Series, rs3: float, rs6: float) -> float:
    """0-100 technical alignment score; not a probability of profit."""
    close = close.dropna()
    if len(close) < 60:
        return np.nan
    sma5 = close.rolling(5).mean().iloc[-1]
    sma10 = close.rolling(10).mean().iloc[-1]
    sma21 = close.rolling(21).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    price = float(close.iloc[-1])
    tests = [price > sma5, price > sma10, price > sma21, price > sma50, sma5 > sma10, sma10 > sma21,
             np.isfinite(rs3) and rs3 > 0, np.isfinite(rs6) and rs6 > 0]
    base = sum(bool(t) for t in tests) / len(tests) * 80
    rsi = calculate_rsi(close).iloc[-1]
    # Reward positive momentum without rewarding extremely overbought RSI.
    rsi_points = 10 if pd.notna(rsi) and 50 <= rsi <= 70 else 5 if pd.notna(rsi) and rsi > 70 else 0
    mom = daily_return(close, 21)
    momentum_points = 10 if np.isfinite(mom) and mom > 0 else 0
    return float(round(min(100, base + rsi_points + momentum_points), 1))

@st.cache_data(ttl=900, show_spinner=False)
def relative_strength_table(kind: str) -> pd.DataFrame:
    if kind == "Sectors":
        universe = SECTORS
    elif kind == "Themes":
        universe = THEMES
    else:
        universe = {symbol: symbol for symbol in STOCKS}
    symbols = sorted(set(universe.values()) | {"SPY"})
    closes = {s: fetch_daily_close(s) for s in symbols}
    rows = []
    for name, ticker in universe.items():
        display_name = STOCKS.get(name, name) if kind == "Stocks" else name
        s_close = closes.get(ticker, pd.Series(dtype=float))
        spy = closes.get("SPY", pd.Series(dtype=float))
        row = {"Name": display_name, "Ticker": ticker}
        for label, days in PERIOD_DAYS.items():
            asset_ret = daily_return(s_close, days)
            spy_ret = daily_return(spy, days)
            row[f"{label} Return %"] = asset_ret
            row[f"{label} vs SPY (pp)"] = asset_ret - spy_ret if np.isfinite(asset_ret) and np.isfinite(spy_ret) else np.nan
        vals = [row["3M vs SPY (pp)"], row["6M vs SPY (pp)"]]
        row["SPY RS Score"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        sp3, sp6 = row["3M vs SPY (pp)"], row["6M vs SPY (pp)"]
        if np.isfinite(sp3) and np.isfinite(sp6):
            row["Relative Classification"] = "BULLISH" if sp3 > 0 and sp6 > 0 else "BEARISH" if sp3 < 0 and sp6 < 0 else "MIXED"
        else:
            row["Relative Classification"] = "INSUFFICIENT DATA"
        row["Signal Quality"] = stock_quality_score(s_close, sp3, sp6) if kind == "Stocks" else np.nan
        row["Data Status"] = "OK" if len(s_close) > 0 and len(spy) > 0 else "MISSING DATA"
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

tabs = st.tabs(["Market Overview", "Top Opportunities", "High RS Stocks", "Sector Rankings", "Theme Rankings", "Stock Scanner", "Forex Scanner", "Commodity Scanner", "Signal History", "Methodology", "🏆 Market Leaders"])

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
    if len(scores) == 3:
        regime = "BULLISH TREND" if overall == "BULLISH" else "BEARISH TREND" if overall == "BEARISH" else "SIDEWAYS / MIXED"
        st.subheader("Market regime")
        st.metric("SPY regime", f"{signal_emoji(overall)} {regime}")
        st.caption("Regime is inferred from SPY's 1-hour, 4-hour, and daily moving-average alignment; it is context, not a forecast.")
    st.caption(f"Data check timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. This is the app check time, not a guarantee that every provider quote is current. The current intraday candle may still be forming.")

with tabs[1]:
    st.subheader("Top Opportunities Across Markets")
    st.caption("Ranks 1–10 show the strongest bullish candidates; ranks 11–20 show the strongest bearish candidates. Stock quality scores combine trend, RSI, momentum, and SPY-relative strength; Forex/commodity scores measure timeframe agreement, so scores are not perfectly comparable across asset classes. Scores are not probabilities of profit.")
    with st.spinner("Building cross-market shortlist..."):
        stocks_all = relative_strength_table("Stocks")
        fx_all = forex_scan_table()
        cmd_all = commodity_scan_table()

    bullish_candidates = []
    bearish_candidates = []

    if not stocks_all.empty:
        for _, r in stocks_all.iterrows():
            cls = str(r.get("Relative Classification", ""))
            quality = r.get("Signal Quality", np.nan)
            if pd.isna(quality):
                continue
            item = {"Market": "Stocks", "Name": r["Name"], "Ticker": r["Ticker"],
                    "Classification": cls, "Signal Quality": float(quality),
                    "SPY RS Score (pp)": r.get("SPY RS Score", np.nan), "Data Status": r.get("Data Status", "Unknown")}
            if cls == "BULLISH":
                # Stock candidates must outperform SPY over both 3M and 6M.
                bullish_candidates.append(item)
            elif cls == "BEARISH":
                # Invert the bullish-oriented stock quality score for a bearish shortlist.
                item["Signal Quality"] = round(100.0 - float(quality), 1)
                bearish_candidates.append(item)

    for market, frame, name_col in [("Forex", fx_all, "Currency Pair"), ("Commodities", cmd_all, "Commodity")]:
        if not frame.empty:
            for _, r in frame.iterrows():
                cls = str(r.get("Relative Classification", ""))
                quality = r.get("Signal Quality / Alignment", np.nan)
                if pd.isna(quality):
                    continue
                item = {"Market": market, "Name": r[name_col], "Ticker": r["Ticker"],
                        "Classification": cls, "Signal Quality": float(quality),
                        "SPY RS Score (pp)": np.nan, "Data Status": r.get("Data Status", "Unknown")}
                if "BULLISH" in cls:
                    bullish_candidates.append(item)
                elif "BEARISH" in cls:
                    bearish_candidates.append(item)

    bullish = pd.DataFrame(bullish_candidates)
    bearish = pd.DataFrame(bearish_candidates)

    st.markdown("<div style='background:#d9f2e3;color:#146c43;padding:10px 14px;border-radius:8px;font-weight:700;font-size:1.1rem'>🟢 BULLISH OPPORTUNITIES — RANKS 1–10</div>", unsafe_allow_html=True)
    if bullish.empty:
        st.info("No bullish candidates with sufficient data are available right now. Try refreshing the scan later.")
    else:
        bullish = bullish.sort_values(["Signal Quality", "SPY RS Score (pp)"], ascending=[False, False], na_position="last").head(10).reset_index(drop=True)
        bullish.insert(0, "Rank", np.arange(1, len(bullish) + 1))
        bullish_style = bullish.style.apply(lambda row: ["background-color: #e6f4ea; color: #14532d; font-weight: 600"] * len(row), axis=1)
        st.dataframe(bullish_style, use_container_width=True, hide_index=True)
        st.download_button("Download bullish opportunities CSV", bullish.to_csv(index=False), "baldeh_top_bullish_opportunities.csv", "text/csv")

    st.markdown("<div style='background:#f8d7da;color:#842029;padding:10px 14px;border-radius:8px;font-weight:700;font-size:1.1rem;margin-top:14px'>🔴 BEARISH OPPORTUNITIES — RANKS 11–20</div>", unsafe_allow_html=True)
    st.caption("Bearish assets are ranked separately from bullish assets. For stocks, more negative 3M/6M relative strength helps break ties; Forex and commodities are ranked by multi-timeframe agreement. This is a watchlist, not an instruction to short.")
    if bearish.empty:
        st.info("No bearish candidates with sufficient data are available right now. Try refreshing the scan later.")
    else:
        bearish = bearish.sort_values(["Signal Quality", "SPY RS Score (pp)"], ascending=[False, True], na_position="last").head(10).reset_index(drop=True)
        bearish.insert(0, "Rank", np.arange(11, 11 + len(bearish)))
        bearish_style = bearish.style.apply(lambda row: ["background-color: #fce8e6; color: #7f1d1d; font-weight: 600"] * len(row), axis=1)
        st.dataframe(bearish_style, use_container_width=True, hide_index=True)
        st.download_button("Download bearish opportunities CSV", bearish.to_csv(index=False), "baldeh_top_bearish_opportunities.csv", "text/csv")

    st.caption("Stocks are classified bullish/bearish only when their 3-month and 6-month returns versus SPY agree. Forex and commodity signals require multi-timeframe agreement. Market conditions and data availability can change after refresh.")

with tabs[2]:
    st.subheader("High Relative Strength Stocks vs SPY")
    st.caption("High RS means the stock has outperformed SPY across both 3-month and 6-month windows. RS is relative performance, not a buy recommendation.")
    min_rs = st.slider("Minimum average 3M/6M excess return versus SPY (percentage points)", min_value=0, max_value=30, value=0, step=1)
    with st.spinner("Finding stocks with high relative strength..."):
        highrs = relative_strength_table("Stocks")
    if highrs.empty:
        st.error("No stock data returned. Try refreshing later.")
    else:
        highrs = highrs[(highrs["Relative Classification"] == "BULLISH") & (highrs["SPY RS Score"] >= min_rs)]
        # Re-rank on every scan refresh by Signal Quality first, then use SPY relative strength as a tie-breaker.
        highrs["Signal Quality"] = pd.to_numeric(highrs["Signal Quality"], errors="coerce")
        highrs = highrs.sort_values(
            ["Signal Quality", "SPY RS Score"],
            ascending=[False, False],
            na_position="last",
        ).copy()
        if not highrs.empty:
            highrs["Rank"] = np.arange(1, len(highrs) + 1)
        if highrs.empty:
            st.info("No stocks currently meet this high-RS threshold in the available data.")
        else:
            highrs_view = highrs[["Rank", "Name", "Ticker", "SPY RS Score", "3M vs SPY (pp)", "6M vs SPY (pp)", "Signal Quality"]].rename(columns={"SPY RS Score": "Avg 3M/6M RS (pp)"})
            st.dataframe(highrs_view.style.format({"Avg 3M/6M RS (pp)": "{:+.2f}", "3M vs SPY (pp)": "{:+.2f}", "6M vs SPY (pp)": "{:+.2f}", "Signal Quality": "{:.0f}/100"}), use_container_width=True, hide_index=True)
            st.download_button("Download high-RS stocks CSV", highrs_view.to_csv(index=False), "baldeh_high_rs_stocks.csv", "text/csv")

with tabs[3]:
    st.subheader("US sector relative strength")
    with st.spinner("Calculating sector rankings..."):
        df = relative_strength_table("Sectors")
    if df.empty:
        st.error("No sector data returned. Try refreshing later.")
    else:
        st.caption("Ranked by relative performance versus SPY. Only rank, name, ticker, and classification are shown.")
        sector_view = df[["Rank", "Name", "Ticker", "Relative Classification"]]
        st.dataframe(color_classification(sector_view), use_container_width=True, hide_index=True)
        st.download_button("Download sector rankings CSV", sector_view.to_csv(index=False), "baldeh_sector_rankings.csv", "text/csv")

with tabs[4]:
    st.subheader("Investment theme relative strength")
    with st.spinner("Calculating theme rankings..."):
        df = relative_strength_table("Themes")
    if df.empty:
        st.error("No theme data returned. Try refreshing later.")
    else:
        st.caption("Ranked by the average 3-month and 6-month relative performance versus SPY. Only rank, name, ticker, and classification are shown.")
        theme_view = df[["Rank", "Name", "Ticker", "Relative Classification"]]
        st.dataframe(color_classification(theme_view), use_container_width=True, hide_index=True)
        st.download_button("Download theme rankings CSV", theme_view.to_csv(index=False), "baldeh_theme_rankings.csv", "text/csv")

with tabs[5]:
    st.subheader("Individual stock scanner")
    search = st.text_input("Filter by ticker or company", placeholder="e.g. NVDA or NVIDIA")
    with st.spinner("Calculating stock rankings..."):
        df = relative_strength_table("Stocks")
    if not df.empty:
        if search.strip():
            mask = df["Ticker"].str.contains(search.strip(), case=False, na=False) | df["Name"].str.contains(search.strip(), case=False, na=False)
            df = df[mask]
        # Sort by strongest Signal Quality on every rerun/refresh; SPY relative strength breaks ties.
        df["Signal Quality"] = pd.to_numeric(df["Signal Quality"], errors="coerce")
        df["SPY RS Score"] = pd.to_numeric(df["SPY RS Score"], errors="coerce")
        df = df.sort_values(["Signal Quality", "SPY RS Score"], ascending=[False, False], na_position="last")
        st.caption("Sorted strongest-to-weakest by Signal Quality, with SPY relative strength as the tie-breaker. The stock universe is curated, not exhaustive. Rankings are research signals, not buy/sell instructions.")
        # Keep the established three-column stock scanner display.
        stock_view = df[["Name", "Ticker", "Relative Classification"]].rename(columns={"Name": "Stock Name"})
        st.dataframe(color_classification(stock_view), use_container_width=True, hide_index=True)
        st.download_button("Download stock classifications CSV", stock_view.to_csv(index=False), "baldeh_stock_classifications.csv", "text/csv")
    else:
        st.error("No stock data returned. Try refreshing later.")

with tabs[6]:
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
        # Highest timeframe-alignment score first; missing scores remain at the bottom.
        fx["Signal Quality / Alignment"] = pd.to_numeric(fx["Signal Quality / Alignment"], errors="coerce")
        fx = fx.sort_values("Signal Quality / Alignment", ascending=False, na_position="last", kind="stable").reset_index(drop=True)
        st.caption("Sorted from strongest to weakest by Signal Quality / Alignment (highest first).")
        st.dataframe(color_classification(fx), use_container_width=True, hide_index=True)
        st.download_button("Download forex scan CSV", fx.to_csv(index=False), "baldeh_forex_scan.csv", "text/csv")

with tabs[7]:
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
        # Highest timeframe-alignment score first; missing scores remain at the bottom.
        commodities["Signal Quality / Alignment"] = pd.to_numeric(commodities["Signal Quality / Alignment"], errors="coerce")
        commodities = commodities.sort_values("Signal Quality / Alignment", ascending=False, na_position="last", kind="stable").reset_index(drop=True)
        st.caption("Sorted from strongest to weakest by Signal Quality / Alignment (highest first).")
        st.dataframe(color_classification(commodities), use_container_width=True, hide_index=True)
        st.download_button("Download commodity scan CSV", commodities.to_csv(index=False), "baldeh_commodity_scan.csv", "text/csv")

with tabs[8]:
    st.subheader("Signal History & Performance Tracker")
    st.caption("Record dated snapshots to compare what signals did next. This is a manual research journal, not an automated backtest. History is held in the current app session and can be downloaded; save the CSV if you want to keep it across sessions or redeployments.")
    if "signal_history" not in st.session_state:
        st.session_state["signal_history"] = []
    if st.button("Record current scan snapshot", key="record_signal_snapshot"):
        snapshot_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        snapshot_rows = []
        try:
            stocks_snap = relative_strength_table("Stocks")
            for _, r in stocks_snap.iterrows():
                px = fetch_daily_close(r["Ticker"])
                snapshot_rows.append({"Snapshot UTC": snapshot_time, "Market": "Stocks", "Name": r["Name"], "Ticker": r["Ticker"],
                    "Signal": r["Relative Classification"], "Price": float(px.iloc[-1]) if len(px) else np.nan,
                    "SPY RS Score (pp)": r["SPY RS Score"], "Signal Quality": r["Signal Quality"]})
            for market, frame, name_col in [("Forex", forex_scan_table(), "Currency Pair"), ("Commodities", commodity_scan_table(), "Commodity")]:
                for _, r in frame.iterrows():
                    snapshot_rows.append({"Snapshot UTC": snapshot_time, "Market": market, "Name": r[name_col], "Ticker": r["Ticker"],
                        "Signal": r["Relative Classification"], "Price": r["Last Price"], "SPY RS Score (pp)": np.nan,
                        "Signal Quality": r["Signal Quality / Alignment"]})
            st.session_state["signal_history"].extend(snapshot_rows)
            st.success(f"Recorded {len(snapshot_rows)} assets at {snapshot_time}.")
        except Exception as exc:
            st.error(f"Snapshot could not be completed: {exc}")
    if st.session_state["signal_history"]:
        hist = pd.DataFrame(st.session_state["signal_history"])
        st.dataframe(hist.tail(100), use_container_width=True, hide_index=True)
        st.download_button("Download signal history CSV", hist.to_csv(index=False), "baldeh_signal_history.csv", "text/csv")
        # Compare the first and latest recorded prices for assets with repeated observations.
        usable = hist.dropna(subset=["Price"]).sort_values("Snapshot UTC")
        if not usable.empty:
            perf_rows = []
            for (market, ticker), grp in usable.groupby(["Market", "Ticker"]):
                if len(grp) >= 2:
                    first, last = grp.iloc[0], grp.iloc[-1]
                    if first["Price"] != 0:
                        perf_rows.append({"Market": market, "Name": last["Name"], "Ticker": ticker,
                            "First Signal": first["Signal"], "Latest Signal": last["Signal"],
                            "Price Change Since First Snapshot %": (last["Price"] / first["Price"] - 1) * 100,
                            "Snapshots": len(grp)})
            if perf_rows:
                perf = pd.DataFrame(perf_rows).sort_values("Price Change Since First Snapshot %", ascending=False)
                st.subheader("Observed price change between your saved snapshots")
                st.dataframe(perf, use_container_width=True, hide_index=True)
                st.caption("This comparison is only meaningful when snapshots span a useful time interval. It is not a backtest and does not include fees, slippage, or trade execution.")
    else:
        st.info("No snapshots recorded in this session yet. Run a scan, then tap 'Record current scan snapshot'. Download the CSV to preserve the journal outside this session.")

with tabs[9]:
    st.subheader("Methodology and limitations")
    st.markdown("""
**Benchmark**
- **SPY:** S&P 500 ETF benchmark. All stock, sector, and theme comparisons use SPY only.

**Relative strength**
- Each sector, theme ETF, or stock's total return is measured over 1, 3, 6, and 12 months.
- Relative strength versus a benchmark is asset return minus benchmark return, expressed in percentage points.
- The **SPY RS Score** averages the 3-month and 6-month relative returns versus SPY. It is a ranking metric, not a forecast.
- **High RS Stocks** filters for stocks positive versus SPY across both 3-month and 6-month periods, then ranks by average excess return.
- **Signal Quality** is a 0-100 rule-alignment score using trend structure, relative strength, RSI, and recent momentum. It is not a probability of profit.
- **BULLISH** means both 3-month and 6-month relative returns beat SPY; **BEARISH** means both underperform SPY; **MIXED** means the periods disagree or are not both positive/negative.

**Trend indicators**
- Market benchmarks use 5 SMA, 10 SMA, 21 SMA, and 20 HMA on 1-hour, 4-hour, and daily candles.
- Four-hour bars are aggregated from hourly bars; session boundaries and incomplete candles can affect intraday readings.

**Forex**
- Forex pairs use 1-hour, 4-hour (aggregated from hourly bars), and daily candles with EMA 20, EMA 50, RSI(14), and 10-candle momentum.
- Multi-timeframe classifications require agreement from available timeframes; disagreement is shown as mixed. Signal quality measures timeframe alignment, not probability of profit.
- Forex data availability and intraday history are controlled by Yahoo Finance.

**Commodities**
- Commodity futures use 1-hour, 4-hour (aggregated from hourly bars), and daily candles with EMA 20, EMA 50, RSI(14), and 10-candle momentum.
- Multi-timeframe classifications require agreement from available timeframes; disagreement is shown as mixed. Futures symbols may have limited or unavailable intraday data, and continuous futures prices are not the same as a specific tradable contract.

**Data**
- Prices are obtained from Yahoo Finance through `yfinance`; availability, timestamps, and historical coverage are controlled by the provider.
- The app caches results briefly to reduce repeated requests. Use **Refresh market data** to clear the cache and request new values.
- Scanner tables include a Data Status column where applicable. The app shows a check timestamp, which is not a guarantee that every provider quote is current.
- If data is missing or stale, verify the provider's response before relying on the reading.
- The Signal History tab records manual snapshots in the current session. Download the CSV to preserve the journal; app sessions may reset after reconnects or redeployments.

**Important**
This tool is for market research and education, not financial advice. Signals are rule-based and may be wrong. Always independently verify data and manage risk.
""")
    st.write("No email-alert code, SMTP settings, or scheduled email workflow is included in this project.")

with tabs[10]:
    st.subheader("🏆 Market Leaders & Best Opportunities")
    st.markdown(
        "<div style='background:linear-gradient(110deg,#102a43,#14532d);color:white;padding:16px 18px;border-radius:12px;margin-bottom:12px'>"
        "<div style='font-size:1.35rem;font-weight:800'>ONE RANKED LIST • THREE MARKETS</div>"
        "<div style='font-size:0.98rem;margin-top:5px'>Stocks · Forex · Commodities — strongest directional signals first</div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Rankings rebuild whenever the app reruns and refresh automatically after you use Refresh market data. Signal Score is a rule-alignment ranking aid—not a probability of profit—and stock, forex, and commodity methods are not perfectly comparable.")
    if st.button("🔄 Refresh all market leaders", key="refresh_market_leaders"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Combining stock, forex, and commodity signals into one leaderboard..."):
        leader_stocks = relative_strength_table("Stocks")
        leader_fx = forex_scan_table()
        leader_cmd = commodity_scan_table()

    leader_rows = []
    if not leader_stocks.empty:
        for _, r in leader_stocks.iterrows():
            classification = str(r.get("Relative Classification", "INSUFFICIENT DATA"))
            quality = pd.to_numeric(pd.Series([r.get("Signal Quality", np.nan)]), errors="coerce").iloc[0]
            if classification == "INSUFFICIENT DATA" or pd.isna(quality):
                continue
            # For stocks, Signal Quality is bullish-oriented; invert it for bearish classifications.
            score = float(quality) if classification == "BULLISH" else 100.0 - float(quality) if classification == "BEARISH" else 0.0
            rs3, rs6 = r.get("3M vs SPY (pp)", np.nan), r.get("6M vs SPY (pp)", np.nan)
            details = f"3M vs SPY: {rs3:+.2f} pp · 6M vs SPY: {rs6:+.2f} pp" if pd.notna(rs3) and pd.notna(rs6) else "SPY relative strength unavailable"
            leader_rows.append({"Market": "Stocks", "Asset": r["Name"], "Ticker": r["Ticker"],
                "Signal": classification, "Signal Score / 100": round(score, 1), "Signal Detail": details,
                "Data Status": r.get("Data Status", "Unknown")})

    for market_name, frame, asset_col in [("Forex", leader_fx, "Currency Pair"), ("Commodities", leader_cmd, "Commodity")]:
        if frame.empty:
            continue
        for _, r in frame.iterrows():
            classification = str(r.get("Relative Classification", "INSUFFICIENT DATA"))
            quality = pd.to_numeric(pd.Series([r.get("Signal Quality / Alignment", np.nan)]), errors="coerce").iloc[0]
            if classification == "INSUFFICIENT DATA" or pd.isna(quality):
                continue
            if "BULLISH" in classification or "BEARISH" in classification:
                score = float(quality)
            else:
                # Mixed timeframe direction is deliberately deprioritized in this directional leaderboard.
                score = 0.0
            details = f"1H: {r.get('1-Hour Signal', '—')} · 4H: {r.get('4-Hour Signal', '—')} · Daily: {r.get('Daily Signal', '—')}"
            leader_rows.append({"Market": market_name, "Asset": r[asset_col], "Ticker": r["Ticker"],
                "Signal": classification, "Signal Score / 100": round(score, 1), "Signal Detail": details,
                "Data Status": r.get("Data Status", "Unknown")})

    leaders = pd.DataFrame(leader_rows)
    if leaders.empty:
        st.warning("No assets with sufficient signal data are available right now. Try refreshing the scan later.")
    else:
        leaders["Signal Score / 100"] = pd.to_numeric(leaders["Signal Score / 100"], errors="coerce")
        leaders["_direction_priority"] = leaders["Signal"].map(lambda x: 0 if "BULLISH" in x or "BEARISH" in x else 1)
        leaders = leaders.sort_values(["Signal Score / 100", "_direction_priority", "Market", "Asset"],
                                      ascending=[False, True, True, True], kind="stable", na_position="last").reset_index(drop=True)
        leaders.insert(0, "Rank", np.arange(1, len(leaders) + 1))
        leaders_display = leaders.drop(columns=["_direction_priority"])

        bullish_count = int(leaders_display["Signal"].str.contains("BULLISH", na=False).sum())
        bearish_count = int(leaders_display["Signal"].str.contains("BEARISH", na=False).sum())
        mixed_count = int(leaders_display["Signal"].str.contains("MIXED", na=False).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Assets Ranked", len(leaders_display))
        m2.metric("🟢 Bullish", bullish_count)
        m3.metric("🔴 Bearish", bearish_count)
        m4.metric("🟡 Mixed", mixed_count)

        st.markdown("#### 🌟 Top 3 ranked signals")
        top3 = leaders_display.head(3)
        cards = st.columns(3)
        for idx, (_, r) in enumerate(top3.iterrows()):
            signal = str(r["Signal"])
            if "BULLISH" in signal:
                bg, fg, accent, icon = "#e6f4ea", "#14532d", "#16834a", "🟢"
            elif "BEARISH" in signal:
                bg, fg, accent, icon = "#fce8e6", "#7f1d1d", "#c62828", "🔴"
            else:
                bg, fg, accent, icon = "#fff3cd", "#856404", "#d39e00", "🟡"
            asset = str(r["Asset"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            market = str(r["Market"])
            ticker = str(r["Ticker"])
            score = float(r["Signal Score / 100"])
            cards[idx].markdown(
                f"<div style='background:{bg};color:{fg};border-top:5px solid {accent};border-radius:12px;padding:14px;min-height:138px'>"
                f"<div style='font-size:.86rem;font-weight:700'>#{int(r['Rank'])} · {icon} {market}</div>"
                f"<div style='font-size:1.12rem;font-weight:800;margin:7px 0 3px'>{asset}</div>"
                f"<div style='font-size:.88rem'>{ticker}</div>"
                f"<div style='font-size:1rem;font-weight:800;margin-top:8px'>Signal Score: {score:.0f}/100</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📊 All-market leaderboard")
        f1, f2, f3 = st.columns(3)
        market_filter = f1.selectbox("Market", ["All markets", "Stocks", "Forex", "Commodities"], key="leaders_market_filter")
        signal_filter = f2.selectbox("Signal type", ["All signals", "BULLISH", "BEARISH", "MIXED"], key="leaders_signal_filter")
        min_score = f3.slider("Minimum signal score", 0, 100, 0, 5, key="leaders_min_score")
        shown = leaders_display.copy()
        if market_filter != "All markets":
            shown = shown[shown["Market"] == market_filter]
        if signal_filter != "All signals":
            shown = shown[shown["Signal"].str.contains(signal_filter, na=False)]
        shown = shown[shown["Signal Score / 100"] >= min_score].copy()

        def paint_leader_rows(row):
            sig = str(row.get("Signal", ""))
            if "BULLISH" in sig:
                color = "background-color: #e6f4ea; color: #14532d; font-weight: 600"
            elif "BEARISH" in sig:
                color = "background-color: #fce8e6; color: #7f1d1d; font-weight: 600"
            elif "MIXED" in sig:
                color = "background-color: #fff3cd; color: #856404; font-weight: 600"
            else:
                color = ""
            return [color] * len(row)

        if shown.empty:
            st.info("No leaderboard entries match those filters. Try lowering the minimum score or changing the filters.")
        else:
            shown = shown.copy()
            st.download_button("⬇️ Download market leaders CSV", shown.to_csv(index=False),
                               "baldeh_market_leaders_best_opportunities.csv", "text/csv")
            display_leaders = shown.copy()
            display_leaders["Signal Score / 100"] = display_leaders["Signal Score / 100"].map(lambda x: f"{x:.0f}")
            st.dataframe(display_leaders.style.apply(paint_leader_rows, axis=1), use_container_width=True, hide_index=True)

        st.caption("Score interpretation: stocks use Signal Quality oriented to the stock's SPY-relative classification; forex and commodities use timeframe alignment for bullish/bearish signals. Mixed signals receive a score of zero so they appear lower in this directional leaderboard. This is a research ranking—not a buy/sell instruction.")

