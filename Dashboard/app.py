"""
Investment Management Dashboard
--------------------------------
Dark-mode Streamlit dashboard: 15 NSE stocks (Pharmaceuticals/IT & Technology/Energy & Power) +
equity portfolio allocation driven by trailing two-month adjusted-price returns + CAPM / MPT / APT / Fama-French reference calculators.

Data: Yahoo Finance via yfinance (.NS suffix).
Caching: st.cache_data with TTLs; synthetic fallbacks on rate-limit / holiday.
Manual ROCE/ROE/PEG calculation from financial statements when .info is empty.

Portfolio return engine: trailing two-month adjusted-price performance; DCF valuation is separate research context.
Risk engine: Standard deviation (annualized) of daily returns — beta is NOT used
             for portfolio allocation. Factor-model tabs remain for reference only.

Run:  streamlit run app.py
"""

# ====================================================================== #
# 0. IMPORTS
# ====================================================================== #
import hashlib
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = Exception

# ====================================================================== #
# 1. PAGE CONFIG & THEME
# ====================================================================== #
st.set_page_config(
    page_title="Investment Management Dashboard",
    page_icon="📊",
    layout="wide",
)

BG = "#0e1117"
CARD = "#1e293b"
ACCENT = "#38bdf8"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
PLOTLY_CFG = {"displaylogo": False, "scrollZoom": True}

YF_TO_PANDAS_FREQ = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "60min", "90m": "90min", "1h": "1h", "1d": "1D", "5d": "5D",
    "1wk": "1W", "1mo": "ME", "3mo": "3ME",
}


def inject_css():
    st.markdown(
        f"""
        <style>
        :root {{
            --bg:#050b14; --panel:#0b1422; --panel2:#0e1a2b; --panel3:#111f31;
            --line:rgba(148,163,184,.14); --line2:rgba(148,163,184,.22);
            --text:#f7fafc; --muted:#8ea0b8; --cyan:#38bdf8; --teal:#2dd4bf;
            --green:#34d399; --red:#fb7185; --gold:#fbbf24; --violet:#a78bfa;
        }}
        .stApp {{
            background:
                radial-gradient(900px 380px at 92% -6%, rgba(56,189,248,.11), transparent 58%),
                radial-gradient(720px 360px at 0% 20%, rgba(45,212,191,.055), transparent 60%),
                linear-gradient(180deg,#050b14 0%,#07101c 52%,#050b14 100%);
            color:var(--text);
        }}
        .main .block-container {{ max-width:1720px; padding:1.15rem 1.7rem 3.5rem; }}
        #MainMenu, footer, header {{ visibility:hidden; }}
        section[data-testid="stSidebar"] {{
            background:linear-gradient(180deg,#06101c 0%,#081321 55%,#07111c 100%);
            border-right:1px solid var(--line);
            min-width:310px;
        }}
        section[data-testid="stSidebar"] .block-container {{ padding:1rem 1rem 2rem; }}
        section[data-testid="stSidebar"] label {{ color:#a9b8cb !important; font-size:.78rem; font-weight:650; }}
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:#0a1523 !important; }}
        h1,h2,h3,h4,h5,h6 {{ color:var(--text) !important; letter-spacing:-.025em; }}
        p,span,div,button,input,textarea,select {{ font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
        .stCaption,[data-testid="stCaptionContainer"] {{ color:#7e90a8 !important; }}
        .brand {{ display:flex; align-items:center; gap:13px; margin:2px 0 20px; }}
        .brand-mark {{
            width:46px;height:46px;border-radius:15px;display:flex;align-items:center;justify-content:center;
            background:linear-gradient(145deg,rgba(56,189,248,.22),rgba(45,212,191,.11));
            border:1px solid rgba(56,189,248,.28); box-shadow:0 12px 30px rgba(0,0,0,.3); font-size:1.25rem;
        }}
        .brand-title {{ font-size:1.05rem;font-weight:800;color:#f8fafc;line-height:1.12; }}
        .brand-sub {{ color:#73859d;font-size:.62rem;margin-top:5px;letter-spacing:.13em;text-transform:uppercase; }}
        .hero {{
            position:relative; overflow:hidden; border:1px solid var(--line2); border-radius:24px;
            padding:26px 30px 24px; margin:0 0 15px;
            background:linear-gradient(135deg,rgba(15,29,47,.98),rgba(7,16,28,.96));
            box-shadow:0 22px 65px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.025);
        }}
        .hero::before {{
            content:""; position:absolute; left:0;top:0;right:0;height:2px;
            background:linear-gradient(90deg,var(--cyan),var(--teal),transparent 70%); opacity:.8;
        }}
        .hero::after {{
            content:""; position:absolute; right:-85px; top:-110px; width:300px; height:300px;
            background:radial-gradient(circle,rgba(56,189,248,.18),transparent 66%); pointer-events:none;
        }}
        .hero-kicker {{ color:#7dd3fc;font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.16em; }}
        .hero h1 {{ margin:.35rem 0 .48rem; font-size:2.2rem; font-weight:850; }}
        .hero p {{ margin:0; color:#9badc2; max-width:1000px; font-size:.88rem; line-height:1.55; }}
        .status-pill {{
            display:inline-flex;align-items:center;gap:8px;margin-top:15px;padding:7px 11px;border-radius:999px;
            font-size:.66rem;font-weight:800;background:rgba(52,211,153,.08);color:#86efac;
            border:1px solid rgba(52,211,153,.18);text-transform:uppercase;letter-spacing:.05em;
        }}
        .status-dot {{ width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 10px #34d399; }}
        .chart-state {{
            display:inline-flex; align-items:center; padding:7px 12px; margin:0 0 13px;
            border:1px solid rgba(56,189,248,.18); border-radius:999px;
            background:rgba(8,20,34,.82); color:#9db0c7; font-size:.67rem; font-weight:750;
            letter-spacing:.04em; text-transform:uppercase;
        }}
        .ticker-strip {{ display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:0 0 16px; }}
        .ticker-card {{
            background:linear-gradient(180deg,rgba(13,27,44,.94),rgba(9,19,32,.94));
            border:1px solid var(--line);border-radius:15px;padding:13px 15px;
            box-shadow:0 12px 30px rgba(0,0,0,.14);
        }}
        .ticker-label {{ color:#74869e;font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800; }}
        .ticker-value {{ color:#f8fafc;font-size:1.15rem;font-weight:850;margin-top:5px; }}
        .ticker-meta {{ color:#7f90a7;font-size:.67rem;margin-top:3px; }}
        .neon-marquee-wrap {{
            position:relative; overflow:hidden; margin:0 0 18px; padding:9px 0;
            border:1px solid rgba(56,189,248,.16); border-radius:14px;
            background:linear-gradient(90deg,rgba(5,12,22,.92),rgba(8,18,31,.76),rgba(5,12,22,.92));
            box-shadow:0 0 24px rgba(56,189,248,.06), inset 0 0 22px rgba(45,212,191,.035);
        }}
        .neon-marquee::before,.neon-marquee::after {{
            content:""; position:absolute; top:0; bottom:0; width:80px; z-index:2; pointer-events:none;
        }}
        .neon-marquee::before {{ left:0; background:linear-gradient(90deg,#07101c,transparent); }}
        .neon-marquee::after {{ right:0; background:linear-gradient(270deg,#07101c,transparent); }}
        .neon-track {{
            display:inline-flex; align-items:center; width:max-content; gap:24px;
            padding-left:24px; animation:neon-scroll 34s linear infinite; will-change:transform;
        }}
        .neon-index-item {{
            display:inline-flex; align-items:center; gap:9px; white-space:nowrap;
            padding:5px 10px; border-radius:999px; background:rgba(10,24,40,.62);
            border:1px solid rgba(56,189,248,.14); box-shadow:0 0 12px rgba(56,189,248,.06);
        }}
        .neon-index-name {{
            color:#67e8f9; font-size:.68rem; font-weight:850; letter-spacing:.06em;
            text-shadow:0 0 8px rgba(103,232,249,.7);
        }}
        .neon-index-price {{
            color:#c4b5fd; font-size:.72rem; font-weight:850;
            text-shadow:0 0 10px rgba(196,181,253,.55);
        }}
        @keyframes neon-scroll {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
        .section-head {{ display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin:22px 0 11px; }}
        .section-head h3 {{ margin:0;font-size:1.05rem; }}
        .section-head p {{ margin:3px 0 0;color:#8191a8;font-size:.72rem; }}
        div[data-testid="stMetric"] {{
            background:linear-gradient(180deg,rgba(15,31,50,.97),rgba(10,20,34,.97));
            border:1px solid var(--line);border-radius:15px;padding:14px 15px;
            box-shadow:0 12px 32px rgba(0,0,0,.15);min-height:96px;
        }}
        div[data-testid="stMetricLabel"] {{ color:#7f90a7 !important;font-size:.65rem !important;text-transform:uppercase;letter-spacing:.07em;font-weight:750; }}
        div[data-testid="stMetricValue"] {{ color:#f8fafc !important;font-weight:850;font-size:1.34rem !important; }}
        div[data-testid="stMetricDelta"] {{ font-size:.68rem !important; }}
        .metric-card {{
            background:linear-gradient(180deg,rgba(16,32,51,.96),rgba(10,20,34,.96));
            border:1px solid var(--line);border-radius:15px;padding:15px;box-shadow:0 12px 30px rgba(0,0,0,.15);height:100%;
        }}
        .metric-label {{ color:#8191a8;font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800; }}
        .metric-value {{ margin-top:6px;color:#f8fafc;font-size:1.35rem;font-weight:850; }}
        .metric-meta {{ margin-top:3px;color:#708198;font-size:.68rem; }}
        .stTabs [data-baseweb="tab-list"] {{
            gap:5px;padding:5px;background:rgba(5,13,23,.88);border:1px solid var(--line);
            border-radius:15px;margin-bottom:17px;box-shadow:0 10px 28px rgba(0,0,0,.16);
            position:sticky;top:7px;z-index:5;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius:10px;padding:9px 13px;color:#7f90a7;background:transparent;font-weight:750;font-size:.72rem;
        }}
        .stTabs [aria-selected="true"] {{
            color:#e0f2fe !important;background:linear-gradient(180deg,rgba(56,189,248,.13),rgba(56,189,248,.055)) !important;
            box-shadow:inset 0 0 0 1px rgba(56,189,248,.2),0 6px 18px rgba(0,0,0,.12);
        }}
        div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,.stNumberInput > div > div,.stTextInput > div > div {{
            background:#091522 !important;border:1px solid var(--line) !important;border-radius:10px !important;
        }}
        .stButton > button {{ border-radius:10px;border:1px solid rgba(56,189,248,.2);background:rgba(56,189,248,.07);color:#dcf7ff;font-weight:750; }}
        .stButton > button:hover {{ border-color:rgba(56,189,248,.4);background:rgba(56,189,248,.12); }}
        .stExpander {{ background:rgba(11,24,40,.74);border:1px solid var(--line);border-radius:14px; }}
        hr {{ border:0;border-top:1px solid var(--line);margin:1rem 0; }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 12px 32px rgba(0,0,0,.12); }}
        .small-note {{ color:#6f829b;font-size:.66rem;line-height:1.5; }}
        @media(max-width:1000px) {{
            .main .block-container {{ padding:1rem .8rem 2rem; }}
            .hero {{ padding:22px; }} .hero h1 {{ font-size:1.65rem; }}
            .ticker-strip {{ grid-template-columns:repeat(2,1fr); }}
        }}
        @media(max-width:600px) {{ .ticker-strip {{ grid-template-columns:1fr; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand():
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div>
                <div class="brand-title">Investment Intelligence</div>
                <div class="brand-sub">Equity research • portfolio analytics • valuation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(selected_name, selected_ticker, sector, is_demo=False):
    status = "Demo fallback active" if is_demo else "Live market data connected"
    status_cls = "" if not is_demo else " style='background:rgba(251,191,36,.08);color:#fde68a;border-color:rgba(251,191,36,.2);'"
    dot = "#fbbf24" if is_demo else "#22c55e"
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Selected research universe • {sector}</div>
            <h1>{selected_name}</h1>
            <p>{selected_ticker} · Institutional-style view across fundamentals, technicals, peer benchmarking, factor models and DCF valuation.</p>
            <div class="status-pill"{status_cls}><span class="status-dot" style="background:{dot};box-shadow:0 0 10px {dot};"></span>{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====================================================================== #
# 2. ASSET UNIVERSE
# ====================================================================== #
SECTORS = {
    "Pharmaceuticals": [
        ("Laurus Labs", "LAURUSLABS.NS"), ("Divi's Laboratories", "DIVISLAB.NS"),
        ("Aurobindo Pharma", "AUROPHARMA.NS"), ("Zydus Lifesciences", "ZYDUSLIFE.NS"),
        ("Granules India", "GRANULES.NS"),
    ],
    "Energy & Power": [
        ("JSW Energy", "JSWENERGY.NS"), ("Torrent Power", "TORNTPOWER.NS"),
        ("NTPC", "NTPC.NS"), ("Adani Power", "ADANIPOWER.NS"), ("Saatvik Green Energy", "SAATVIK.NS"),
    ],
    "IT & Technology": [
        ("Coforge", "COFORGE.NS"), ("Persistent Systems", "PERSISTENT.NS"),
        ("Mphasis", "MPHASIS.NS"), ("Hexaware", "HEXAWARE.NS"), ("KPIT Technologies", "KPIT.NS"),
    ],
}

# Expanded candidate pool used only for the momentum portfolio; other dashboard modules retain their original 15-stock universe.
PORTFOLIO_CANDIDATES = {
    "Pharmaceuticals": [
        ("Laurus Labs", "LAURUSLABS.NS"), ("Divi's Laboratories", "DIVISLAB.NS"),
        ("Aurobindo Pharma", "AUROPHARMA.NS"), ("Zydus Lifesciences", "ZYDUSLIFE.NS"),
        ("Granules India", "GRANULES.NS"), ("Sai Life Sciences", "SAILIFE.NS"),
        ("Wockhardt", "WOCKPHARMA.NS"), ("Marksans Pharma", "MARKSANS.NS"),
        ("Gland Pharma", "GLAND.NS"), ("Jubilant Pharmova", "JUBLPHARMA.NS"),
        ("Caplin Point Laboratories", "CAPLIPOINT.NS"), ("Neuland Laboratories", "NEULANDLAB.NS"),
        ("Alembic Pharmaceuticals", "APLLTD.NS"), ("Natco Pharma", "NATCOPHARM.NS"),
    ],
    "Energy & Power": [
        ("JSW Energy", "JSWENERGY.NS"), ("Torrent Power", "TORNTPOWER.NS"),
        ("NTPC", "NTPC.NS"), ("Adani Power", "ADANIPOWER.NS"), ("Saatvik Green Energy", "SAATVIK.NS"),
        ("KPI Green Energy", "KPIGREEN.NS"), ("Suzlon Energy", "SUZLON.NS"),
        ("Inox Wind", "INOXWIND.NS"), ("Waaree Renewable Technologies", "WAAREERTL.NS"),
        ("RattanIndia Power", "RTNPOWER.NS"), ("Reliance Power", "RPOWER.NS"),
        ("Orient Green Power", "GREENPOWER.NS"), ("Websol Energy System", "WEBELSOLAR.NS"),
        ("Transformers & Rectifiers India", "TRIL.NS"),
    ],
    "IT & Technology": [
        ("Coforge", "COFORGE.NS"), ("Persistent Systems", "PERSISTENT.NS"),
        ("Mphasis", "MPHASIS.NS"), ("Hexaware", "HEXAWARE.NS"), ("KPIT Technologies", "KPIT.NS"),
        ("Birlasoft", "BSOFT.NS"), ("Zensar Technologies", "ZENSARTECH.NS"),
        ("Happiest Minds", "HAPPSTMNDS.NS"), ("Newgen Software", "NEWGEN.NS"),
        ("Intellect Design Arena", "INTELLECT.NS"), ("Tanla Platforms", "TANLA.NS"),
        ("RateGain Travel Technologies", "RATEGAIN.NS"), ("Affle India", "AFFLE.NS"),
        ("Sasken Technologies", "SASKEN.NS"),
    ],
}

ALL_TICKERS = {name: t for names in SECTORS.values() for name, t in names}
ALL_TICKER_LIST = list(ALL_TICKERS.values())
ALL_TICKER_NAMES = list(ALL_TICKERS.keys())
ALL_STOCKS = [(name, ticker) for names in SECTORS.values() for name, ticker in names]

INDEX_CANDIDATES = {
    "Nifty 50": ["^NSEI", "^NSE50"],
    "Sensex": ["^BSESN"],
    "Bank Nifty": ["^NSEBANK"],
    "Nifty Midcap": ["^CNXMIDCAP", "^NSEMDCP50", "^NSMIDCP"],
    "Nifty Smallcap": ["^CNXSMALLCAP", "^NSESMALLCAP"],
    "Nifty IT": ["^CNXIT"],
    "Nifty Pharma": ["^CNXPHARMA"],
    "Nifty Energy": ["^CNXENERGY"],
    "Nifty Auto": ["^CNXAUTO"],
    "Nifty Financial Services": ["^CNXFINANCE"],
    "Nifty FMCG": ["^CNXFMCG"],
}

SECTOR_INDEX = {
    "Pharmaceuticals": "Nifty Pharma",
    "IT & Technology": "Nifty IT",
    "Energy & Power": "Nifty Energy",
}

DEFAULT_RF = 6.9


# ====================================================================== #
# 3. SAFE FORMATTERS
# ====================================================================== #
def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fmt_pct(val, decimals=1):
    f = _safe_float(val)
    if f is None:
        return "N/A"
    return f"{f * 100:.{decimals}f}%"


def _fmt_num(val, prefix="", decimals=0):
    f = _safe_float(val)
    if f is None:
        return "N/A"
    if abs(f) >= 1e12:
        return f"{prefix}{f / 1e12:,.{decimals}f}T"
    if abs(f) >= 1e9:
        return f"{prefix}{f / 1e9:,.{decimals}f}B"
    if abs(f) >= 1e7:
        return f"{prefix}{f / 1e7:,.{decimals}f}Cr"
    if abs(f) >= 1e5:
        return f"{prefix}{f / 1e5:,.{decimals}f}L"
    return f"{prefix}{f:,.{decimals}f}"


def _fmt_ratio(val, decimals=2):
    f = _safe_float(val)
    if f is None:
        return "N/A"
    return f"{f:.{decimals}f}"


# ====================================================================== #
# 4. DATA FETCHING
# ====================================================================== #
@st.cache_data(ttl=900, show_spinner=False)
def generate_synthetic_data(ticker, period, interval):
    end = pd.Timestamp.now(tz="Asia/Kolkata")
    mapping = {
        "1d": (1, "5min"), "5d": (5, "15min"), "1mo": (30, "1D"), "3mo": (90, "1D"),
        "6mo": (180, "1D"), "1y": (365, "1D"), "2y": (730, "1D"),
        "5y": (1825, "1D"), "max": (3650, "1D"),
    }
    days, fallback_int = mapping.get(period, (365, "1D"))
    pandas_freq = YF_TO_PANDAS_FREQ.get(fallback_int, fallback_int)
    dates = pd.date_range(end=end, periods=days * 5, freq=pandas_freq, tz="Asia/Kolkata")
    np.random.seed(hash(ticker) % (2**31))
    price = 1000 + np.random.randn(len(dates)).cumsum() * 10
    price = np.maximum(price, 100)
    vol = np.random.randint(100_000, 5_000_000, len(dates)).astype(float)
    df = pd.DataFrame({
        "Open": price + np.random.randn(len(dates)) * 5,
        "High": price + abs(np.random.randn(len(dates)) * 10),
        "Low": price - abs(np.random.randn(len(dates)) * 10),
        "Close": price,
        "Volume": vol,
    }, index=dates)
    df.index.name = "Datetime"
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker, period, interval, allow_demo):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            return df, False
    except (YFRateLimitError, Exception):
        pass
    if allow_demo:
        return generate_synthetic_data(ticker, period, interval), True
    return pd.DataFrame(), False


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker):
    data = {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        data = {
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "trailingPE": info.get("trailingPE"),
            "sectorPE": info.get("sectorPE"),
            "pegRatio": info.get("pegRatio"),
            "bookValue": info.get("bookValue"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnCapitalEmployed": info.get("returnOnCapitalEmployed"),
            "debtToEquity": info.get("debtToEquity"),
            "trailingEps": info.get("trailingEps"),
            "freeCashflow": info.get("freeCashflow"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "shortName": info.get("shortName", ticker),
            "ebitda": info.get("ebitda"),
            "operatingMargin": info.get("operatingMargin"),
            "profitMargin": info.get("profitMargin"),
            "grossMargin": info.get("grossMargin"),
            "payoutRatio": info.get("payoutRatio"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "enterpriseValue": info.get("enterpriseValue"),
            "ebitdaMargins": info.get("ebitdaMargins"),
        }
    except Exception:
        pass

    try:
        tk_obj = yf.Ticker(ticker)
        m_roce, m_roe, m_peg = calc_manual_metrics(tk_obj, data)
        if _safe_float(data.get("returnOnCapitalEmployed")) is None and m_roce is not None:
            data["returnOnCapitalEmployed"] = m_roce
        if _safe_float(data.get("returnOnEquity")) is None and m_roe is not None:
            data["returnOnEquity"] = m_roe
        if _safe_float(data.get("pegRatio")) is None and m_peg is not None:
            data["pegRatio"] = m_peg
    except Exception:
        pass

    return data


def _get_from_statement(df, keys, index=0):
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    for k in keys:
        if k in df.index:
            s = df.loc[k].dropna()
            if len(s) > index:
                try:
                    return float(s.iloc[index])
                except (TypeError, ValueError):
                    continue
    return None


def calc_manual_metrics(ticker_obj, info_dict):
    try:
        fin = ticker_obj.financials
    except Exception:
        fin = None
    try:
        bs = ticker_obj.balance_sheet
    except Exception:
        bs = None

    ebit = _get_from_statement(fin, ["Operating Income", "EBIT"])
    assets = _get_from_statement(bs, ["Total Assets"])
    curr_liab = _get_from_statement(bs, ["Total Current Liabilities", "Current Liabilities"])
    roce = None
    if ebit is not None and assets is not None and curr_liab is not None:
        cap_emp = assets - curr_liab
        if cap_emp > 0:
            roce = ebit / cap_emp

    net_inc = _get_from_statement(fin, ["Net Income", "Net Income Common Stockholders",
                                         "Net Income From Continuing Operation"])
    equity = _get_from_statement(bs, ["Stockholders Equity", "Common Stock Equity",
                                       "Total Equity Gross Minority Interest"])
    roe = None
    if net_inc is not None and equity is not None and equity > 0:
        roe = net_inc / equity

    peg = None
    pe = _safe_float(info_dict.get("trailingPE") if isinstance(info_dict, dict) else None)
    eps_curr = _get_from_statement(fin, ["Diluted EPS", "Basic EPS"], index=0)
    eps_prev = _get_from_statement(fin, ["Diluted EPS", "Basic EPS"], index=1)
    if pe is not None and eps_curr is not None and eps_prev is not None and eps_prev > 0:
        eps_growth = (eps_curr - eps_prev) / abs(eps_prev)
        if eps_growth > 0:
            peg = pe / (eps_growth * 100)

    return roce, roe, peg


@st.cache_data(ttl=86400, show_spinner=False)
def load_index_close(key, period):
    for tk_str in INDEX_CANDIDATES.get(key, []):
        try:
            df = yf.Ticker(tk_str).history(period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index)
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
                return s
        except Exception:
            continue
    return None


@st.cache_data(ttl=60, show_spinner=False)
def load_live_index_price(key):
    """Return the latest intraday/available price for a market index."""
    for tk_str in INDEX_CANDIDATES.get(key, []):
        try:
            df = yf.Ticker(tk_str).history(period="1d", interval="5m", auto_adjust=False)
            if df is not None and not df.empty and "Close" in df:
                close = df["Close"].dropna()
                if len(close):
                    return float(close.iloc[-1])
        except Exception:
            continue
    daily = load_index_close(key, "5d")
    if daily is not None and len(daily):
        try:
            return float(daily.dropna().iloc[-1])
        except Exception:
            pass
    return None


def render_index_marquee():
    """Render a neon scrolling market-index strip."""
    keys = [
        "Nifty 50", "Sensex", "Bank Nifty", "Nifty Midcap", "Nifty Smallcap",
        "Nifty IT", "Nifty Pharma", "Nifty Energy", "Nifty Auto",
        "Nifty Financial Services", "Nifty FMCG",
    ]
    items = []
    for key in keys:
        value = load_live_index_price(key)
        value_text = f"{value:,.2f}" if value is not None else "N/A"
        items.append(
            f'<span class="neon-index-item"><span class="neon-index-name">{key}</span>'
            f'<span class="neon-index-price">{value_text}</span></span>'
        )
    track = "".join(items) + "".join(items)
    st.markdown(
        f"""<div class="neon-marquee-wrap" aria-label="Live market index ticker">
            <div class="neon-marquee"><div class="neon-track">{track}</div></div>
        </div>""", unsafe_allow_html=True,
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_close_series(ticker, period, allow_demo):
    h, is_demo = load_history(ticker, period, "1d", allow_demo)
    if len(h):
        return h["Close"], is_demo
    return None, is_demo


def _flatten_price_columns(history):
    """Normalize yfinance single-level and multi-level OHLC columns."""
    if history is None or history.empty:
        return pd.DataFrame()
    out = history.copy()
    if isinstance(out.columns, pd.MultiIndex):
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
        names = []
        for col in out.columns:
            parts = [str(part) for part in col if part is not None and str(part) not in {"", "None"}]
            names.append(next((part for part in parts if part in known), parts[0] if parts else ""))
        out.columns = names
        if out.columns.duplicated().any():
            out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def load_portfolio_price_snapshot(ticker, lookback_start):
    """Fetch actual daily prices; retry direct download if cached ticker history is incomplete."""
    errors = []
    history = pd.DataFrame()
    try:
        history, _ = load_history(ticker, "3mo", "1d", False)
    except Exception as exc:
        errors.append(f"Ticker.history: {type(exc).__name__}: {exc}")
    history = _flatten_price_columns(history)

    # yfinance versions vary in adjusted-close and column-index behaviour.
    if history.empty or "Close" not in history.columns or "Adj Close" not in history.columns:
        try:
            download_start = (pd.Timestamp(lookback_start) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            try:
                downloaded = yf.download(ticker, start=download_start, interval="1d", auto_adjust=False,
                                         progress=False, threads=False, multi_level_index=False)
            except TypeError:
                downloaded = yf.download(ticker, start=download_start, interval="1d", auto_adjust=False,
                                         progress=False, threads=False)
            downloaded = _flatten_price_columns(downloaded)
            if downloaded is not None and not downloaded.empty:
                if history.empty or "Close" not in history.columns or "Adj Close" in downloaded.columns:
                    history = downloaded
        except Exception as exc:
            errors.append(f"yf.download: {type(exc).__name__}: {exc}")

    history = _flatten_price_columns(history)
    if history.empty or "Close" not in history.columns:
        return None, "; ".join(errors) if errors else "No Close-price series returned"
    try:
        history.index = pd.to_datetime(history.index)
        if getattr(history.index, "tz", None) is not None:
            history.index = history.index.tz_localize(None)
        history = history.sort_index()
    except Exception as exc:
        return None, f"Could not normalize price history: {type(exc).__name__}: {exc}"

    raw_close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if raw_close.empty:
        return None, "; ".join(errors) if errors else "Close-price series is empty"
    adjusted_source = history["Adj Close"] if "Adj Close" in history.columns else history["Close"]
    adjusted = pd.to_numeric(adjusted_source, errors="coerce").dropna()
    if adjusted.empty:
        adjusted = raw_close.copy()
    adjusted.index = pd.to_datetime(adjusted.index)
    raw_close.index = pd.to_datetime(raw_close.index)
    if getattr(adjusted.index, "tz", None) is not None:
        adjusted.index = adjusted.index.tz_localize(None)
    if getattr(raw_close.index, "tz", None) is not None:
        raw_close.index = raw_close.index.tz_localize(None)

    cutoff = pd.Timestamp(lookback_start)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    window = adjusted.loc[adjusted.index >= cutoff]
    return_pct, start_price = np.nan, np.nan
    latest_adjusted = float(adjusted.iloc[-1])
    if len(window) >= 2 and float(window.iloc[0]) > 0:
        start_price = float(window.iloc[0])
        latest_adjusted = float(window.iloc[-1])
        return_pct = (latest_adjusted / start_price - 1.0) * 100.0

    return {
        "Ticker": ticker,
        "2-Month Return %": return_pct,
        "Return Start Price": start_price,
        "Latest Adjusted Price": latest_adjusted,
        "Latest Close Price": float(raw_close.iloc[-1]),
    }, None


# ====================================================================== #
# 5. MATH & INDICATORS
# ====================================================================== #
def calculate_rsi(series, periods=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / periods, min_periods=periods).mean()
    avg_loss = loss.ewm(alpha=1.0 / periods, min_periods=periods).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (tp * df["Volume"]).groupby(df.index.date).cumsum()
    cum_vol = df["Volume"].groupby(df.index.date).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def to_daily_returns(s):
    return s.pct_change().dropna()


def to_monthly(s):
    try:
        return s.resample("ME").last().pct_change().dropna()
    except ValueError:
        return s.resample("M").last().pct_change().dropna()


def ols_regression(y, factors):
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    X = np.column_stack([np.ones(len(factors)), factors.values])
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_arr))
    X, y_arr = X[mask], y_arr[mask]
    if len(X) < factors.shape[1] + 2:
        return 0.0, np.zeros(factors.shape[1]), 0.0
    result, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
    alpha = result[0]
    betas = result[1:]
    y_pred = X @ result
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, betas, r2


# --------------------------------------------------------------------- #
# Risk helper — annualized standard deviation of daily returns
# --------------------------------------------------------------------- #
def annualized_std(close_series, periods_per_year=252):
    """Annualized standard deviation of daily returns (decimal)."""
    if close_series is None or len(close_series) < 30:
        return None
    rets = close_series.pct_change().dropna()
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(periods_per_year))


# ====================================================================== #
# 6. UI RENDERERS
# ====================================================================== #
def render_screener(ticker, fi):
    short = fi.get('shortName', ticker)
    sector = fi.get('sector', 'N/A')
    industry = fi.get('industry', 'N/A')
    st.markdown(f'<div class="section-head"><div><h3>{short}</h3><p>{ticker} · {sector} · {industry}</p></div></div>', unsafe_allow_html=True)

    rows = [
        ("Price", _fmt_num(fi.get("currentPrice"), prefix="₹"), "Market snapshot"),
        ("Market Cap", _fmt_num(fi.get("marketCap"), prefix="₹"), "Equity value"),
        ("52W High", _fmt_num(fi.get("fiftyTwoWeekHigh"), prefix="₹"), "52-week range"),
        ("52W Low", _fmt_num(fi.get("fiftyTwoWeekLow"), prefix="₹"), "52-week range"),
    ]
    cols = st.columns(4)
    for c, (label, value, meta) in zip(cols, rows):
        with c:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-meta">{meta}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><h3>Valuation</h3><p>Multiple-based snapshot</p></div></div>', unsafe_allow_html=True)
    vals = [
        ("P/E", _fmt_ratio(fi.get("trailingPE")), "TTM"),
        ("Sector P/E", _fmt_ratio(fi.get("sectorPE"), 1), "Benchmark"),
        ("PEG", _fmt_ratio(fi.get("pegRatio")), "Growth-adjusted"),
        ("P/B", _fmt_ratio(fi.get("priceToBook")), "Book multiple"),
    ]
    cols = st.columns(4)
    for c, (label, value, meta) in zip(cols, vals):
        with c:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-meta">{meta}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><h3>Profitability & returns</h3><p>Operating quality and capital efficiency</p></div></div>', unsafe_allow_html=True)
    de_raw = _safe_float(fi.get("debtToEquity"))
    prof = [
        ("Dividend Yield", _fmt_pct(fi.get("dividendYield")), "Shareholder yield"),
        ("ROCE", _fmt_pct(fi.get("returnOnCapitalEmployed")), "Capital efficiency"),
        ("ROE", _fmt_pct(fi.get("returnOnEquity")), "Equity efficiency"),
        ("D/E", _fmt_ratio(de_raw / 100 if de_raw is not None else None), "Leverage"),
    ]
    cols = st.columns(4)
    for c, (label, value, meta) in zip(cols, prof):
        with c:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-meta">{meta}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><h3>Earnings & cash</h3><p>Per-share and free-cash-flow indicators</p></div></div>', unsafe_allow_html=True)
    cash = [
        ("EPS (TTM)", _fmt_num(fi.get("trailingEps"), prefix="₹"), "Trailing earnings"),
        ("Book Value", _fmt_num(fi.get("bookValue"), prefix="₹"), "Per share"),
        ("Free Cash Flow", _fmt_num(fi.get("freeCashflow"), prefix="₹"), "Latest available"),
    ]
    cols = st.columns(3)
    for c, (label, value, meta) in zip(cols, cash):
        with c:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-meta">{meta}</div></div>', unsafe_allow_html=True)


def render_charts(ticker, df_hist, interval="1d", period="1y"):
    if df_hist.empty:
        st.warning("No price data available.")
        return
    df = df_hist.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    try:
        df["VWAP"] = calculate_vwap(df)
    except Exception:
        df["VWAP"] = np.nan

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20], subplot_titles=("Price", "Volume", "RSI"),
    )
    colors_up, colors_dn = GREEN, RED
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_fillcolor=colors_up, decreasing_fillcolor=colors_dn,
    ), row=1, col=1)

    for col, label, color in [
        ("SMA20", "SMA 20", ACCENT),
        ("SMA50", "SMA 50", AMBER),
        ("SMA200", "SMA 200", "#a855f7"),
    ]:
        if df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label,
                                     line=dict(width=1, color=color)), row=1, col=1)

    if df["VWAP"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                                 line=dict(width=1, color="white", dash="dot")), row=1, col=1)

    vol_colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.6), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(width=1, color=ACCENT)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=RED, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=750,
        showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=40, b=30), hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)


def render_compare(sector, tickers):
    st.subheader(f"📊 {sector} — Peer Comparison")
    rows = []
    hist_map = {}
    for name, ticker in tickers:
        fi = load_fundamentals(ticker)
        h, _ = load_history(ticker, "1y", "1d", False)
        ret = None
        if not h.empty and len(h) > 1:
            ret = (h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100
            hist_map[name] = h["Close"]

        ebitda = _safe_float(fi.get("ebitda"))
        op_margin = _safe_float(fi.get("operatingMargin"))
        net_margin = _safe_float(fi.get("profitMargin"))
        roce = _safe_float(fi.get("returnOnCapitalEmployed"))
        roe = _safe_float(fi.get("returnOnEquity"))
        de = _safe_float(fi.get("debtToEquity"))
        if de is not None:
            de = de / 100

        rows.append({
            "Stock": name,
            "P/E": _safe_float(fi.get("trailingPE")),
            "PEG": _safe_float(fi.get("pegRatio")),
            "Mkt Cap (Cr)": _safe_float(fi.get("marketCap")),
            "1Y Return %": _safe_float(ret),
            "EBITDA (Cr)": ebitda / 1e7 if ebitda is not None else None,
            "EBITDA %": op_margin * 100 if op_margin is not None else None,
            "Net Margin %": net_margin * 100 if net_margin is not None else None,
            "ROCE %": roce * 100 if roce is not None else None,
            "ROE %": roe * 100 if roe is not None else None,
            "D/E": de,
        })

    df_cmp = pd.DataFrame(rows)
    avg_row = {}
    for col in df_cmp.columns:
        if col == "Stock":
            avg_row[col] = "Sector Average"
        else:
            vals = df_cmp[col].dropna()
            avg_row[col] = vals.mean() if len(vals) > 0 else None
    df_cmp = pd.concat([df_cmp, pd.DataFrame([avg_row])], ignore_index=True)

    fmt_dict = {
        "P/E": "{:.1f}", "PEG": "{:.2f}", "Mkt Cap (Cr)": "{:,.0f}",
        "1Y Return %": "{:.1f}%", "EBITDA (Cr)": "{:,.0f}", "EBITDA %": "{:.1f}%",
        "Net Margin %": "{:.1f}%", "ROCE %": "{:.1f}%", "ROE %": "{:.1f}%", "D/E": "{:.2f}",
    }
    st.dataframe(
        df_cmp.style.format(fmt_dict, na_rep='N/A')
               .apply(lambda x: ['background-color: #2a3a5a' if x.name == len(df_cmp)-1 else '' for _ in x], axis=1),
        use_container_width=True, hide_index=True,
    )

    if hist_map:
        st.subheader("Relative Performance (Normalized to 100)")
        normed = {}
        for name, s in hist_map.items():
            if s is not None and len(s) > 0:
                normed[name] = (s / s.iloc[0]) * 100
        if normed:
            df_norm = pd.DataFrame(normed)
            fig = px.line(df_norm, template="plotly_dark",
                          labels={"value": "Normalized Price", "index": "Date"})
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=400,
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)


def render_factor_models(ticker, sector, all_stocks):
    """CAPM / MPT / APT / Fama-French — reference only; not used in allocation."""
    st.subheader("🧮 Factor Models & Portfolio Analysis")
    st.caption("These tabs describe systematic risk factors for reference; the portfolio expected-return calculation uses trailing two-month adjusted-price performance.")

    tab_capm, tab_mpt, tab_apt, tab_ff = st.tabs(["CAPM", "MPT", "APT", "Fama-French 3F"])
    rf = st.slider("Risk-free rate (%)", 0.0, 15.0, DEFAULT_RF, 0.1, key="rf_main")

    stock_close, _ = load_close_series(ticker, "2y", True)
    nifty_close = load_index_close("Nifty 50", "2y")

    if stock_close is None:
        st.error("Could not load price data for factor analysis.")
        return

    stock_ret_m = to_monthly(stock_close)
    nifty_ret_m = to_monthly(nifty_close) if nifty_close is not None else None

    with tab_capm:
        st.markdown("**Capital Asset Pricing Model**: $E[R] = R_f + \\beta \\times (E[R_m] - R_f)$")
        st.caption("Reference only — this beta is not fed into the allocation engine.")
        if nifty_ret_m is None:
            st.warning("Nifty 50 index data unavailable — CAPM requires a market proxy.")
            beta_manual = st.number_input("Manual Beta", 0.0, 3.0, 1.0, 0.05, key="beta_manual")
            er_stock = rf / 100 + beta_manual * 0.07
            st.metric("E[R] (CAPM with manual beta)", f"{er_stock * 100:.2f}%")
        else:
            aligned = pd.concat([stock_ret_m, nifty_ret_m], axis=1, join="inner").dropna()
            aligned.columns = ["Stock", "Market"]
            rf_m = (rf / 100) / 12
            excess_stock = aligned["Stock"] - rf_m
            excess_mkt = aligned["Market"] - rf_m
            alpha_m, betas, r2 = ols_regression(excess_stock, excess_mkt)
            beta = betas[0]
            alpha_ann = alpha_m * 12
            er_mkt = excess_mkt.mean() * 12
            er_stock = rf / 100 + beta * er_mkt

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Beta", f"{beta:.3f}")
            mc2.metric("Alpha (annual)", f"{alpha_ann * 100:.2f}%")
            mc3.metric("R²", f"{r2:.4f}")
            mc4.metric("E[R] (CAPM)", f"{er_stock * 100:.2f}%")

            fig = px.scatter(x=excess_mkt, y=excess_stock, trendline="ols",
                             labels={"x": "Market Excess Return", "y": "Stock Excess Return"},
                             template="plotly_dark", title=f"CAPM Regression — {ticker}")
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=400)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    with tab_mpt:
        st.markdown("**Modern Portfolio Theory** — Monte Carlo Efficient Frontier")
        st.caption("Select any stocks from the list below (across all sectors).")
        all_options = [f"{name} ({ticker})" for name, ticker in all_stocks]
        default_selected = [f"{name} ({ticker})" for name, t in all_stocks if t == ticker]
        if len(default_selected) == 0 and len(all_options) > 0:
            default_selected = [all_options[0]]
        if len(all_options) > 1 and len(default_selected) == 1:
            for opt in all_options:
                if opt != default_selected[0]:
                    default_selected.append(opt)
                    break

        selected = st.multiselect("Select stocks for portfolio (min 2)", options=all_options,
                                  default=default_selected[:2], key="mpt_all")
        if len(selected) < 2:
            st.info("Please select at least 2 stocks.")
        else:
            mpt_tickers = [s.split("(")[-1].replace(")", "").strip() for s in selected]
            n_sims = st.slider("Simulations", 1000, 20000, 5000, 500, key="mpt_sims")

            closes = {}
            for tk in mpt_tickers:
                s, _ = load_close_series(tk, "1y", True)
                if s is not None:
                    closes[tk] = s
            df_closes = pd.DataFrame(closes).dropna()
            if len(df_closes) < 30:
                st.warning("Insufficient overlapping data for these stocks.")
            else:
                rets = df_closes.pct_change().dropna()
                mean_ret = rets.mean() * 12
                cov = rets.cov() * 12
                n_assets = len(mpt_tickers)
                rf_ann = rf / 100

                rng = np.random.default_rng(42)
                weights = rng.dirichlet(np.ones(n_assets), size=n_sims)
                port_ret = weights @ mean_ret.values
                port_vol = np.sqrt(np.sum(weights * (weights @ cov.values), axis=1))
                sharpe = (port_ret - rf_ann) / np.where(port_vol > 0, port_vol, 1e-9)

                idx_sharpe = np.argmax(sharpe)
                idx_minvol = np.argmin(port_vol)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port_vol, y=port_ret, mode="markers",
                                         marker=dict(size=2, color=sharpe, colorscale="Viridis", showscale=True),
                                         text=[f"Sharpe: {s:.2f}" for s in sharpe], name="Portfolios"))
                fig.add_trace(go.Scatter(x=[port_vol[idx_sharpe]], y=[port_ret[idx_sharpe]],
                                         marker=dict(size=14, symbol="star", color=GREEN), name="Max Sharpe"))
                fig.add_trace(go.Scatter(x=[port_vol[idx_minvol]], y=[port_ret[idx_minvol]],
                                         marker=dict(size=14, symbol="star", color=RED), name="Min Volatility"))
                for i, tk in enumerate(mpt_tickers):
                    fig.add_trace(go.Scatter(x=[np.sqrt(cov.values[i, i])], y=[mean_ret.values[i]],
                                             marker=dict(size=10, symbol="diamond", color=AMBER),
                                             name=tk.split(".")[0]))

                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                                  xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
                                  height=500, title="Efficient Frontier (Monte Carlo)")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

                st.subheader("Optimal Portfolios")
                w_cols = st.columns(2)
                for col, label, idx in [(w_cols[0], "⭐ Max Sharpe", idx_sharpe),
                                        (w_cols[1], "🛡️ Min Volatility", idx_minvol)]:
                    with col:
                        st.markdown(f"**{label}**")
                        w_df = pd.DataFrame({"Stock": mpt_tickers,
                                             "Weight": [f"{w:.1%}" for w in weights[idx]]})
                        st.dataframe(w_df, hide_index=True, use_container_width=True)
                        st.metric("Return", f"{port_ret[idx]*100:.1f}%")
                        st.metric("Volatility", f"{port_vol[idx]*100:.1f}%")
                        st.metric("Sharpe", f"{sharpe[idx]:.2f}")

    with tab_apt:
        st.markdown("**Arbitrage Pricing Theory** — Multi-Factor Model")
        sector_idx_key = SECTOR_INDEX.get(sector, "Nifty 50")
        factor_names = ["Market (Nifty 50)", "Size (SC−Nifty)", f"Sector ({sector_idx_key})"]

        lambda_m = st.slider("λ Market premium (%)", 0.0, 15.0, 7.0, 0.5, key="apt_lm")
        lambda_s = st.slider("λ Size premium (%)", 0.0, 10.0, 3.0, 0.5, key="apt_ls")
        lambda_sec = st.slider("λ Sector premium (%)", 0.0, 10.0, 2.0, 0.5, key="apt_lsec")

        nifty_m = load_index_close("Nifty 50", "2y")
        small_m = load_index_close("Nifty Smallcap", "2y")
        sec_m = load_index_close(sector_idx_key, "2y")

        if any(s is None for s in [nifty_m, small_m, sec_m]):
            st.warning("Some index data missing.")
        else:
            f_mkt = to_monthly(nifty_m)
            f_size = to_monthly(small_m) - to_monthly(nifty_m)
            f_sec = to_monthly(sec_m) - to_monthly(nifty_m)
            factors_df = pd.concat([f_mkt, f_size, f_sec], axis=1, join="inner").dropna()
            factors_df.columns = ["Market", "Size", "Sector"]
            rf_m = (rf / 100) / 12
            aligned_s = stock_ret_m.reindex(factors_df.index).dropna()
            common = factors_df.index.intersection(aligned_s.index)
            y = aligned_s.loc[common] - rf_m
            X = factors_df.loc[common]
            alpha_a, betas_a, r2_a = ols_regression(y, X)
            er_apt = rf / 100 + betas_a[0] * lambda_m / 100 + betas_a[1] * lambda_s / 100 + betas_a[2] * lambda_sec / 100
            contributions = betas_a * np.array([lambda_m, lambda_s, lambda_sec]) / 100

            factor_cols = st.columns(3)
            for i, fname in enumerate(factor_names):
                factor_cols[i % 3].metric(f"β {fname}", f"{betas_a[i]:.3f}")
            summary_cols = st.columns(4)
            summary_cols[0].metric("R²", f"{r2_a:.4f}")
            summary_cols[1].metric("Alpha (monthly)", f"{alpha_a * 100:.3f}%")
            summary_cols[2].metric("E[R] APT", f"{er_apt * 100:.2f}%")
            summary_cols[3].metric("Contribution", f"{sum(contributions) * 100:.2f}%")
            st.bar_chart(pd.Series(contributions * 100, index=factor_names,
                                   name="Factor Contribution (%)"), template="plotly_dark")

    with tab_ff:
        st.markdown("**Fama-French 3-Factor Model**")
        use_manual_ff = st.checkbox("Use manual factor returns for Fama-French", value=False)
        if use_manual_ff:
            col_mkt, col_smb, col_hml = st.columns(3)
            with col_mkt:
                mkt_ret_ff = st.number_input("MKT monthly return (%)", value=1.0, step=0.1) / 100
            with col_smb:
                smb_ret = st.number_input("SMB monthly return (%)", value=0.4, step=0.1) / 100
            with col_hml:
                hml_ret = st.number_input("HML monthly return (%)", value=0.3, step=0.1) / 100
            er_ff_manual = rf/100 + mkt_ret_ff + smb_ret + hml_ret
            st.metric("E[R] FF3 (manual)", f"{er_ff_manual * 100:.2f}%")
        else:
            upload = st.file_uploader("Upload custom factor CSV (optional)", ["csv"], key="ff_csv")
            nifty_m2 = load_index_close("Nifty 50", "2y")
            mid_m = load_index_close("Nifty Midcap", "2y")
            small_m2 = load_index_close("Nifty Smallcap", "2y")
            if upload is not None:
                custom = pd.read_csv(upload, index_col=0, parse_dates=True)
                custom.columns = ["MKT", "SMB", "HML"]
                ff_factors = to_monthly(custom["MKT"]).to_frame()
                ff_factors["SMB"] = to_monthly(custom["SMB"])
                ff_factors["HML"] = to_monthly(custom["HML"])
                st.success("Custom factor data loaded.")
            elif any(s is None for s in [nifty_m2, mid_m, small_m2]):
                st.warning("Index data unavailable.")
                ff_factors = None
            else:
                ff_factors = pd.DataFrame(index=to_monthly(nifty_m2).index)
                ff_factors["MKT"] = to_monthly(nifty_m2)
                ff_factors["SMB"] = to_monthly(small_m2) - to_monthly(nifty_m2)
                ff_factors["HML"] = to_monthly(mid_m) - to_monthly(nifty_m2)

            if ff_factors is not None:
                ff_factors = ff_factors.dropna()
                rf_m = (rf / 100) / 12
                aligned_s = stock_ret_m.reindex(ff_factors.index).dropna()
                common = ff_factors.index.intersection(aligned_s.index)
                y = aligned_s.loc[common] - rf_m
                X = ff_factors.loc[common]
                alpha_ff, betas_ff, r2_ff = ols_regression(y, X)
                er_ff = rf / 100 + betas_ff[0] * ff_factors["MKT"].mean() * 12 + \
                        betas_ff[1] * ff_factors["SMB"].mean() * 12 + \
                        betas_ff[2] * ff_factors["HML"].mean() * 12

                ff_cols = st.columns(3)
                ff_cols[0].metric("β MKT", f"{betas_ff[0]:.3f}")
                ff_cols[1].metric("β SMB", f"{betas_ff[1]:.3f}")
                ff_cols[2].metric("β HML", f"{betas_ff[2]:.3f}")

                ff_sum = st.columns(3)
                ff_sum[0].metric("R²", f"{r2_ff:.4f}")
                ff_sum[1].metric("Alpha (monthly)", f"{alpha_ff * 100:.3f}%")
                ff_sum[2].metric("E[R] FF3", f"{er_ff * 100:.2f}%")

                st.bar_chart(
                    pd.Series([betas_ff[0] * ff_factors["MKT"].mean() * 12,
                               betas_ff[1] * ff_factors["SMB"].mean() * 12,
                               betas_ff[2] * ff_factors["HML"].mean() * 12],
                              index=["MKT", "SMB", "HML"], name="Factor Contribution (%)"),
                    template="plotly_dark")


# ====================================================================== #
# 7. VALUATION FUNCTIONS (DCF-first)
# ====================================================================== #
@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_statements(ticker):
    tk = yf.Ticker(ticker)
    try:
        income = tk.financials
    except Exception:
        income = pd.DataFrame()
    try:
        balance = tk.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = pd.DataFrame()
    return income, balance, cashflow


def get_item_from_df(df, keys, index=0):
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            s = df.loc[k].dropna()
            if len(s) > index:
                try:
                    return float(s.iloc[index])
                except (TypeError, ValueError):
                    continue
    return None


def compute_free_cash_flow(ticker, income, balance, cashflow):
    net_income = get_item_from_df(income, ["Net Income", "Net Income Common Stockholders"])
    income_tax = get_item_from_df(income, ["Income Tax Expense", "Tax Provision"])
    tax_rate = 0.25
    if net_income is not None and income_tax is not None and net_income != 0:
        pre_tax = net_income + income_tax
        if pre_tax != 0:
            tax_rate = income_tax / pre_tax

    ebit = get_item_from_df(income, ["EBIT", "Operating Income"])
    depreciation = get_item_from_df(cashflow, ["Depreciation", "Depreciation And Amortization"])
    capex = get_item_from_df(cashflow, ["Capital Expenditure", "Purchase Of Property Plant Equipment"])

    delta_wc = 0.0
    if not balance.empty:
        ca_series = balance.loc["Total Current Assets"].dropna() if "Total Current Assets" in balance.index else None
        cl_series = balance.loc["Total Current Liabilities"].dropna() if "Total Current Liabilities" in balance.index else None
        if ca_series is not None and len(ca_series) >= 2 and cl_series is not None and len(cl_series) >= 2:
            ca_curr, ca_prev = float(ca_series.iloc[0]), float(ca_series.iloc[1])
            cl_curr, cl_prev = float(cl_series.iloc[0]), float(cl_series.iloc[1])
            delta_wc = (ca_curr - ca_prev) - (cl_curr - cl_prev)

    fcff = None
    if ebit is not None and depreciation is not None and capex is not None:
        fcff = ebit * (1 - tax_rate) + depreciation - capex - delta_wc
    elif net_income is not None and depreciation is not None and capex is not None:
        fcff = net_income + depreciation - capex - delta_wc

    interest_exp = None
    debt_change = None
    fcfe = None

    if fcff is not None:
        interest_exp = get_item_from_df(income, ["Interest Expense", "Interest Paid"])
        interest_after_tax = interest_exp * (1 - tax_rate) if interest_exp is not None else 0
        debt_series = balance.loc["Total Debt"].dropna() if "Total Debt" in balance.index else None
        debt_change = 0
        if debt_series is not None and len(debt_series) >= 2:
            debt_change = float(debt_series.iloc[0]) - float(debt_series.iloc[1])
        fcfe = fcff - interest_after_tax + debt_change

    return {
        "fcff": fcff, "fcfe": fcfe, "ebit": ebit, "net_income": net_income,
        "tax_rate": tax_rate, "depreciation": depreciation, "capex": capex,
        "delta_wc": delta_wc, "interest_exp": interest_exp, "debt_change": debt_change,
    }


def compute_sector_average(tickers):
    pe_vals, pb_vals, peg_vals, ev_ebitda_vals = [], [], [], []
    for name, t in tickers:
        fi = load_fundamentals(t)
        pe = _safe_float(fi.get("trailingPE"))
        pb = _safe_float(fi.get("priceToBook"))
        peg = _safe_float(fi.get("pegRatio"))
        ev = _safe_float(fi.get("enterpriseValue"))
        ebitda = _safe_float(fi.get("ebitda"))
        if pe is not None: pe_vals.append(pe)
        if pb is not None: pb_vals.append(pb)
        if peg is not None: peg_vals.append(peg)
        if ev is not None and ebitda is not None and ebitda != 0:
            ev_ebitda_vals.append(ev / ebitda)
    avg_pe = np.mean(pe_vals) if pe_vals else None
    avg_pb = np.mean(pb_vals) if pb_vals else None
    avg_peg = np.mean(peg_vals) if peg_vals else None
    avg_ev_ebitda = np.mean(ev_ebitda_vals) if ev_ebitda_vals else None
    return avg_pe, avg_pb, avg_peg, avg_ev_ebitda


# NOTE: intentionally NOT decorated with @st.cache_data.
# Passing a list-of-tuples (sector_tickers) can fail to hash on older Streamlit
# versions and silently drop the entire Portfolio Allocation tab. Individual
# sub-calls (load_fundamentals, load_financial_statements) remain cached.
def compute_stock_valuation(
    ticker, sector_tickers, rf, wacc, cost_equity,
    growth_ddm, growth_fcf, growth_terminal,
):
    fi = load_fundamentals(ticker)
    current_price = _safe_float(fi.get("currentPrice"))
    eps = _safe_float(fi.get("trailingEps"))
    book_value = _safe_float(fi.get("bookValue"))
    pe = _safe_float(fi.get("trailingPE"))
    pb = _safe_float(fi.get("priceToBook"))
    peg = _safe_float(fi.get("pegRatio"))
    sector_pe = _safe_float(fi.get("sectorPE"))

    shares_raw = _safe_float(fi.get("sharesOutstanding"))
    if shares_raw and shares_raw > 0:
        shares_m = shares_raw / 1e6
    else:
        mcap = _safe_float(fi.get("marketCap"))
        px = _safe_float(fi.get("currentPrice"))
        shares_m = (mcap / px) / 1e6 if (mcap and px and px > 0) else None

    div_yield = _safe_float(fi.get("dividendYield"))
    payout_ratio = _safe_float(fi.get("payoutRatio"))
    d0 = None
    if div_yield is not None and current_price is not None:
        d0 = div_yield * current_price
    elif payout_ratio is not None and eps is not None:
        d0 = payout_ratio * eps

    ddm_value = None
    if d0 is not None and d0 > 0:
        required_return = cost_equity if cost_equity > 0 else (rf + 0.05)
        if required_return > growth_ddm:
            ddm_value = d0 * (1 + growth_ddm) / (required_return - growth_ddm)

    income, balance, cashflow = load_financial_statements(ticker)
    cf_data = compute_free_cash_flow(ticker, income, balance, cashflow)
    fcff = cf_data.get("fcff")
    fcfe = cf_data.get("fcfe")

    fcff_value = None
    fcfe_value = None

    if shares_m is not None and shares_m > 0:
        if fcff is not None:
            fcff_proj = [fcff * (1 + growth_fcf) ** i for i in range(1, 6)]
            tv_fcff = (fcff_proj[-1] * (1 + growth_terminal) / (wacc - growth_terminal)
                       if wacc > growth_terminal else 0.0)
            fcff_pv = sum(f / (1 + wacc) ** i for i, f in enumerate(fcff_proj, start=1))
            if tv_fcff > 0:
                fcff_pv += tv_fcff / (1 + wacc) ** 5

            total_debt = get_item_from_df(balance, ["Total Debt", "Long Term Debt", "Short Term Debt"])
            cash_equiv = get_item_from_df(balance, ["Cash And Cash Equivalents", "Cash"])
            equity_value = (fcff_pv - total_debt + cash_equiv
                            if (total_debt is not None and cash_equiv is not None) else fcff_pv)
            fcff_value = equity_value / (shares_m * 1e6)

        if fcfe is not None:
            fcfe_proj = [fcfe * (1 + growth_fcf) ** i for i in range(1, 6)]
            tv_fcfe = (fcfe_proj[-1] * (1 + growth_terminal) / (cost_equity - growth_terminal)
                       if cost_equity > growth_terminal else 0.0)
            fcfe_pv = sum(f / (1 + cost_equity) ** i for i, f in enumerate(fcfe_proj, start=1))
            if tv_fcfe > 0:
                fcfe_pv += tv_fcfe / (1 + cost_equity) ** 5
            fcfe_value = fcfe_pv / (shares_m * 1e6)

    avg_pe, avg_pb, avg_peg, avg_ev_ebitda = compute_sector_average(sector_tickers)
    implied_pe = eps * avg_pe if (eps is not None and avg_pe is not None) else None
    implied_pb = book_value * avg_pb if (book_value is not None and avg_pb is not None) else None

    implied_ev_ebitda = None
    ebitda = _safe_float(fi.get("ebitda"))
    if avg_ev_ebitda is not None and ebitda is not None and shares_m:
        ev_implied = ebitda * avg_ev_ebitda
        debt = get_item_from_df(balance, ["Total Debt", "Long Term Debt", "Short Term Debt"])
        cash = get_item_from_df(balance, ["Cash And Cash Equivalents", "Cash"])
        if debt is not None and cash is not None:
            equity_implied = ev_implied - debt + cash
            implied_ev_ebitda = equity_implied / (shares_m * 1e6)

    implied_sector_pe = eps * sector_pe if (eps is not None and sector_pe is not None) else None

    ev = _safe_float(fi.get("enterpriseValue"))
    ev_ebitda_ratio = (ev / ebitda) if (ev is not None and ebitda) else None

    return {
        "current_price": current_price, "pe": pe, "pb": pb, "peg": peg,
        "ev_ebitda": ev_ebitda_ratio, "ddm": ddm_value, "fcff": fcff_value, "fcfe": fcfe_value,
        "implied_pe": implied_pe, "implied_pb": implied_pb,
        "implied_ev_ebitda": implied_ev_ebitda, "implied_sector_pe": implied_sector_pe,
    }


def compute_dcf_target_price(val_row, current_price, winsor_mult=2.5):
    """
    Blend DCF and relative-valuation outputs into one target price.

    Keys must match the DataFrame row exactly:
      "FCFF", "FCFE", "DDM",
      "Implied (Sector P/E)", "Implied (Sector P/B)", "Implied (Sector EV/EBITDA)"
    """
    if current_price is None or current_price <= 0:
        return current_price, 0

    entries = []

    def _add(value, weight):
        v = _safe_float(value)
        if v is not None and v > 0:
            entries.append((min(v, current_price * winsor_mult), weight))

    _add(val_row.get("FCFF"),                        0.40)
    _add(val_row.get("FCFE"),                        0.25)
    _add(val_row.get("DDM"),                         0.10)
    _add(val_row.get("Implied (Sector P/E)"),        0.15)
    _add(val_row.get("Implied (Sector P/B)"),        0.05)
    _add(val_row.get("Implied (Sector EV/EBITDA)"),  0.05)

    if not entries:
        return current_price, 0

    total_w = sum(w for _, w in entries)
    target = sum(v * w for v, w in entries) / total_w
    return float(target), len(entries)


@st.cache_data(ttl=604800, show_spinner=False)
def weekly_return_table(ticker_map, monday_anchor):
    rows = []
    for sector, names in ticker_map.items():
        for name, ticker in names:
            close, _ = load_close_series(ticker, "3mo", False)
            if close is None or len(close) < 10:
                continue
            s = close.copy()
            idx = pd.to_datetime(s.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            s.index = idx.normalize()
            weekly = s.resample("W-FRI").agg(["first", "last"]).dropna()
            weekly["Return %"] = (weekly["last"] / weekly["first"] - 1) * 100
            weekly["Week Ending"] = weekly.index

            anchor = pd.Timestamp(monday_anchor)
            current_week_end = anchor + pd.Timedelta(days=4)
            completed = weekly[weekly["Week Ending"] < current_week_end].tail(5)

            for _, r in completed.iterrows():
                rows.append({
                    "Sector": sector, "Stock": name, "Ticker": ticker,
                    "Week Ending": r["Week Ending"].date(),
                    "Weekly Return %": r["Return %"],
                })
    return pd.DataFrame(rows)


def render_weekly_analysis():
    st.subheader("📅 Weekly Analysis — Last 5 Completed Weeks")
    today = pd.Timestamp.now().normalize()
    monday = today - pd.to_timedelta(today.weekday(), unit="D")
    st.caption(f"Dynamic Monday refresh • 5 most recent completed weeks as of {today.date()} "
               f"(refresh key: {monday.date()}).")

    df_weekly = weekly_return_table(SECTORS, monday.date().isoformat())
    if df_weekly.empty:
        st.warning("Weekly return data is unavailable right now.")
        return

    wide = df_weekly.pivot_table(index=["Sector", "Stock"], columns="Week Ending",
                                 values="Weekly Return %", aggfunc="first").reset_index()
    week_cols = sorted([c for c in wide.columns if c not in ["Sector", "Stock"]], reverse=True)[:5]
    wide = wide[["Sector", "Stock"] + week_cols]

    st.dataframe(wide.style.format({c: "{:.2f}%" for c in week_cols}, na_rep="N/A"),
                 use_container_width=True, hide_index=True)

    sector_week = df_weekly.groupby(["Sector", "Week Ending"], as_index=False)["Weekly Return %"].mean()
    sector_pivot = sector_week.pivot(index="Sector", columns="Week Ending",
                                     values="Weekly Return %").reset_index()
    sec_cols = sorted([c for c in sector_pivot.columns if c != "Sector"], reverse=True)[:5]
    st.markdown("#### Sector Weekly Average")
    st.dataframe(sector_pivot[["Sector"] + sec_cols]
                 .style.format({c: "{:.2f}%" for c in sec_cols}, na_rep="N/A"),
                 use_container_width=True, hide_index=True)


# ====================================================================== #
# 8. PORTFOLIO ALLOCATION — trailing two-month return, historical price based
# ====================================================================== #
def compute_return_tilted_weights(returns_pct, floor=0.03, tilt=1.0, cap=0.15):
    """
    Return-tilted portfolio weights that never fall to zero.

    1. Every stock first receives a guaranteed floor weight.
    2. The remaining capital is distributed in proportion to exp(tilt * z), where z is the
       winsorised z-score of the stock's trailing two-month return. Higher return -> larger
       share. tilt = 0 reproduces equal weights; larger tilt concentrates more heavily.
    3. No stock may exceed `cap`; any excess is redistributed pro-rata to uncapped stocks.

    Stocks with a missing return are treated as average (z = 0) so they are not punished
    for a data gap. Returns a numpy array that sums to 1.0.
    """
    r = pd.to_numeric(pd.Series(returns_pct), errors="coerce").astype(float)
    n = len(r)
    if n == 0:
        return np.array([])
    floor = float(np.clip(floor, 0.0, 0.9 / n))
    cap = float(max(cap, 1.0 / n + 1e-9))
    floor = min(floor, cap)

    valid = r.dropna()
    if len(valid) >= 2 and float(valid.std(ddof=0)) > 1e-9:
        z = ((r - valid.mean()) / valid.std(ddof=0)).clip(-2.0, 2.0).fillna(0.0)
    else:
        z = pd.Series(0.0, index=r.index)
    score = np.exp(float(tilt) * z.values)
    tilt_share = score / score.sum()

    w = floor + (1.0 - n * floor) * tilt_share
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any() or excess <= 0:
            break
        w[free] += excess * (w[free] / w[free].sum())
    return w / w.sum()


# News / macro context as of 15 Oct 2026 (static snapshot — refresh before relying on it).
MACRO_ASOF = "15 Oct 2026"
MACRO_BACKDROP = [
    "RBI raised the repo rate 25 bp to 5.50% on 7 Oct 2026 (first hike since Feb 2023) and moved to a 'calibrated tightening' stance; FY27 CPI forecast 5.2%, GDP forecast 7.1% after Q1 growth of 7.8%.",
    "10-year G-sec yield hit a ~29-month high near 7.1%; higher rates penalise leveraged, capex-heavy businesses and long-duration (high P/E) stocks.",
    "Brent has been above $100 on West Asia / Strait of Hormuz disruption; the rupee set record lows near 96–97/USD in 2026; FPIs have sold roughly ₹3 lakh crore of Indian equities in CY2026.",
    "Net effect: a weak rupee helps USD earners (IT, pharma exporters) while higher oil, yields and FPI selling weigh on domestic cyclical and leveraged names.",
]
SECTOR_THESIS = {
    "IT & Technology": {
        "news": "TCS's Q2 FY27 print (8 Oct) beat the market's checklist — constant-currency growth of 0.5% QoQ, ~24% operating margin and roughly $9.6B of deal wins — and Nifty IT jumped ~3% on 9 Oct. US visa/PERM curbs and AI-driven pricing pressure remain the swing risks.",
        "economy": "Weak rupee lifts USD revenue conversion; domestic rate hikes barely touch asset-light, net-cash IT balance sheets, which makes the sector a relative safe haven while FPIs sell cyclicals.",
        "risk": "Growth is still low single-digit in dollar terms, and one-day rallies have faded before. Mid-caps with a stronger recent return get more weight here, but the floor keeps laggards in the portfolio.",
    },
    "Pharmaceuticals": {
        "news": "On 22 Jul 2026 the US proposed phased tariffs on imported generics: 0% until Aug 2028, then 100%, rising to 200%. Nifty Pharma fell ~2% that day. Separately, the BIOSECURE Act (in the FY26 NDAA) is steering big-pharma outsourcing toward non-China CDMOs.",
        "economy": "A weak rupee flatters export earnings, and tariffs only bite in 2028, leaving a two-year window to localise. Pharma is defensive against oil and rate shocks.",
        "risk": "Stocks with high US-generics dependence carry long-dated tariff risk; CDMO/API names are structurally better placed. Weights therefore tilt on recent returns, but the floor and cap prevent any single tariff headline from dominating.",
    },
    "Energy & Power": {
        "news": "Centrum expects power demand to grow ~6% a year with 45–50 GW of renewable additions annually; peak demand hit 271 GW in May 2026 vs 242 GW for FY26. India added ~17 GW of renewables in Apr–Jul 2026. Grid build-out is lagging renewables, so curtailment is a risk.",
        "economy": "Strong GDP growth (7%+) supports electricity demand and capex, and import-substituting solar/wind manufacturers benefit from domestic demand. Offsetting this, higher yields and FPI outflows raise funding costs for leveraged developers, and high oil hurts gas-based and import-dependent generation.",
        "risk": "Small-cap power/renewable stocks are the most volatile in the basket; returns can reverse quickly, so the cap limits concentration here.",
    },
}
COMPANY_NOTES = {
    "Aurobindo Pharma": "US was ~46% of FY26 revenue — the most tariff-exposed name here — but it already runs three US plants, a head start on localisation.",
    "Divi's Laboratories": "API + custom-synthesis leader; BIOSECURE steers innovator contracts to clean-record Indian CDMOs, while a weak rupee helps realisations.",
    "Laurus Labs": "CDMO is ~22% of revenue and higher-margin than ARV APIs, so mix is shifting toward the BIOSECURE beneficiary segment; CDMO revenue can be lumpy.",
    "Sai Life Sciences": "Pure-play CDMO that benefits from China+1 outsourcing, though it sold off with the sector on the 22 Jul tariff headline.",
    "Neuland Laboratories": "Complex APIs and custom manufacturing sit in the CDMO growth pocket rather than commodity generics.",
    "Zydus Lifesciences": "Meaningful US generics exposure; fell with the sector on the 22 Jul tariff announcement.",
    "NTPC": "Largest thermal base-load generator; rising peak demand supports utilisation and dispatchable power is needed alongside intermittent renewables.",
    "Waaree Renewable Technologies": "Solar manufacturing/EPC benefits from domestic demand and import substitution; Q1 is seasonally weak.",
    "Saatvik Green Energy": "Solar module maker riding domestic demand and import substitution; project timing makes returns volatile.",
    "KPI Green Energy": "Renewable developer/EPC riding the capacity build-out; project funding costs are sensitive to the rate hike.",
    "Suzlon Energy": "Wind OEM benefiting from 45–50 GW/yr renewable additions; order-book execution and working capital are key.",
    "Inox Wind": "Wind OEM exposed to the same renewable build-out; margins depend on execution and funding costs.",
    "Transformers & Rectifiers India": "Transformer maker: grid expansion lagging renewable additions is a direct demand driver.",
    "Coforge": "Mid-cap IT name: carries more growth sensitivity than large-caps if the TCS-led demand recovery broadens (and more downside if it fades).",
    "Persistent Systems": "Mid-cap digital-engineering name that benefits if the TCS-led demand recovery broadens.",
    "KPIT Technologies": "Auto-software specialist, less tied to generic IT-services pricing than diversified peers.",
}


def build_allocation_rationale(df_alloc, n_stocks):
    """One row per stock explaining its weight in terms of return rank, news and macro context."""
    equal_w = 1.0 / max(n_stocks, 1)
    ranked = df_alloc["2-Month Return %"].rank(ascending=False, method="min")
    rows = []
    for idx, r in df_alloc.iterrows():
        w = float(r["Allocation %"])
        ret = _safe_float(r["2-Month Return %"])
        if w > equal_w * 1.15:
            stance = "Overweight"
        elif w < equal_w * 0.85:
            stance = "Underweight (floor held)"
        else:
            stance = "Neutral"
        if ret is None:
            why_ret = "No usable two-month return, so treated as average and held near equal weight."
        else:
            why_ret = f"Ranks #{int(ranked.loc[idx])} of {n_stocks} on 2-month return ({ret:+.1f}%), giving a {w * 100:.1f}% weight vs {equal_w * 100:.1f}% equal-weight."
        thesis = SECTOR_THESIS.get(r["Sector"], {})
        pieces = [why_ret]
        note = COMPANY_NOTES.get(r["Asset"])
        if note:
            pieces.append(note)
        if thesis.get("economy"):
            pieces.append("Macro: " + thesis["economy"].split(". ")[0].rstrip(".") + ".")
        rows.append({"Asset": r["Asset"], "Sector": r["Sector"], "Weight %": w * 100,
                     "Stance": stance, "Rationale": " ".join(pieces)})
    return pd.DataFrame(rows)


def render_universe_valuation():
    st.subheader("📊 Small-Cap Momentum Portfolio — 5 Stocks per Sector")
    st.caption("Selects the five strongest trailing two-month adjusted-price performers from each sector. Portfolio returns, expected gains, and return charts use this historical price performance—not DCF valuation. Historical returns are illustrative and are not a forecast.")

    # ---------------- Weekly analysis (never halt the tab) ---------------- #
    try:
        render_weekly_analysis()
    except Exception as e:
        st.warning(f"Weekly analysis unavailable: {type(e).__name__}: {e}")

    st.divider()

    # ---------------- Historical-return portfolio settings ---------------- #
    st.markdown("### Historical-return portfolio settings")
    rf, wacc, cost_equity = DEFAULT_RF / 100, (DEFAULT_RF + 4.0) / 100, (DEFAULT_RF + 5.0) / 100
    growth_terminal, growth_ddm, growth_fcf = 0.03, 0.05, 0.05
    st.caption("Two-month return = latest adjusted close ÷ adjusted close on or after the date two calendar months earlier − 1. The same adjusted-price series feeds stock selection, portfolio expected return, expected rupee gain, and the return charts.")

    goal_col1, goal_col2 = st.columns(2)
    with goal_col1:
        goal_start = st.number_input("Starting portfolio value (₹ Cr)", 0.10, 100.0, 1.00, 0.05, key="uni_goal_start")
    with goal_col2:
        # Keep both the default and any stored widget value valid when start capital changes.
        goal_end_default = min(max(1.75, float(goal_start)), 250.0)
        try:
            if float(st.session_state.get("uni_goal_end", goal_end_default)) < float(goal_start):
                st.session_state["uni_goal_end"] = goal_end_default
        except Exception:
            pass
        goal_end = st.number_input("2Y target portfolio value (₹ Cr)", min_value=goal_start,
                                    max_value=250.0, value=goal_end_default, step=0.05, key="uni_goal_end")

    required_2y_return = max(goal_end / goal_start - 1.0, 0.0)
    required_cagr = ((goal_end / goal_start) ** 0.5 - 1.0) if goal_start > 0 and goal_end > 0 else 0.0
    st.info(
        f"2-year goal: ₹{goal_start:.2f} Cr → ₹{goal_end:.2f} Cr | "
        f"Required cumulative return: {required_2y_return * 100:.1f}% | "
        f"Required CAGR: {required_cagr * 100:.1f}% p.a."
    )
    st.divider()

    # ---------------- Build price-only candidate universe ---------------- #
    # Do not call DCF/fundamental valuation for all 42 candidates here: allocation
    # expected returns are based only on the real two-month adjusted-close series.
    # The dedicated Valuation tab remains available for DCF and relative multiples.
    all_rows = []
    for sector, tickers in PORTFOLIO_CANDIDATES.items():
        for name, ticker in tickers:
            all_rows.append({
                "Stock": name,
                "Ticker": ticker,
                "Sector": sector,
                "Current Price": np.nan,
            })
    if not all_rows:
        st.error("No portfolio candidates are configured.")
        return
    df = pd.DataFrame(all_rows)

    # Rank sectors using real trailing two-calendar-month adjusted-price returns.
    # Retry via direct download and keep rendering the allocation if any history is absent.
    lookback_start = (pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
                      - pd.DateOffset(months=2))
    momentum_rows, history_errors = [], []
    candidate_count = len(df)
    for _, stock_row in df.iterrows():
        ticker = stock_row["Ticker"]
        try:
            record, error = load_portfolio_price_snapshot(ticker, lookback_start)
            if record is not None:
                momentum_rows.append(record)
            else:
                history_errors.append(f"{stock_row['Stock']} ({ticker}): {error or 'no usable price history'}")
        except Exception as exc:
            history_errors.append(f"{stock_row['Stock']} ({ticker}): {type(exc).__name__}: {exc}")

    momentum_columns = ["Ticker", "2-Month Return %", "Return Start Price", "Latest Adjusted Price", "Latest Close Price"]
    momentum_df = pd.DataFrame(momentum_rows, columns=momentum_columns)
    if momentum_df.empty:
        st.warning(
            "Yahoo Finance did not provide usable daily price history for the candidate stocks. "
            "Portfolio weights, capital allocation and allocation charts will still be shown, "
            "but historical returns will appear as N/A until real price history is available. "
            "No synthetic returns are substituted."
        )
    else:
        valid_return_count = int(pd.to_numeric(momentum_df["2-Month Return %"], errors="coerce").notna().sum())
        st.caption(f"Historical-price coverage: {valid_return_count}/{candidate_count} candidate stocks have usable two-month returns.")
        if valid_return_count < candidate_count:
            st.warning("Some stocks have incomplete two-month price history. Their return fields will show N/A; portfolio allocation will continue.")

    if history_errors:
        with st.expander(f"Price-history diagnostics ({len(history_errors)} candidates without usable close prices)", expanded=False):
            for message in history_errors:
                st.write(message)

    # A left merge prevents the previous inner merge from dropping candidates without history.
    # The previous empty-frame return hid the allocation editor, KPIs, tables, and every chart.
    df = df.merge(momentum_df, on="Ticker", how="left")
    for col in ["Current Price", "Latest Close Price", "Latest Adjusted Price", "2-Month Return %"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Current Price"] = df["Latest Close Price"].combine_first(df["Current Price"]).combine_first(df["Latest Adjusted Price"])
    df = (df.sort_values(["Sector", "2-Month Return %"], ascending=[True, False], na_position="last")
            .groupby("Sector", group_keys=False).head(5).reset_index(drop=True))
    if df.empty:
        st.error("The candidate stock list is empty; no portfolio rows can be rendered.")
        return

    if df["2-Month Return %"].notna().any():
        st.markdown("#### Selected stocks — ranked by actual two-month return where history is available")
    else:
        st.markdown("#### Selected stocks — default universe while waiting for actual two-month price history")
    display_selection = df[["Stock", "Sector", "2-Month Return %", "Return Start Price", "Latest Adjusted Price", "Latest Close Price"]]
    try:
        st.dataframe(
            display_selection.sort_values(["Sector", "2-Month Return %"], ascending=[True, False], na_position="last")
                .style.format({"2-Month Return %": "{:+.2f}%", "Return Start Price": "₹{:.2f}", "Latest Adjusted Price": "₹{:.2f}", "Latest Close Price": "₹{:.2f}"}, na_rep="N/A"),
            use_container_width=True, hide_index=True,
        )
    except Exception:
        st.dataframe(display_selection, use_container_width=True, hide_index=True)

    st.caption("DCF and relative valuation are intentionally not run in the portfolio-allocation path. Expected returns are calculated from actual two-month adjusted closes; use the separate Valuation tab for DCF and fundamental metrics.")

    # ---------------- Investor profile ---------------- #
    st.subheader("💼 Portfolio Allocation & 2-Year Goal")
    st.caption("Capital is allocated across up to 15 selected equities (five per sector). Edit company ratios below; ratios normalize to 100%. Expected returns use actual trailing two-month adjusted-close performance, not DCF. If a return is unavailable, it is shown as N/A rather than fabricated.")

    st.markdown("### Investor Profile & Risk Controls")
    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        investor_type = st.selectbox("Investor type",
            ["Conservative", "Moderate", "Aggressive", "Custom"], index=1, key="uni_investor_type")
    with prof_col2:
        risk_adjustment = st.slider("Risk adjustment", 0.50, 1.50, 1.00, 0.05, key="uni_risk_adjustment",
                                    help="<1 reduces the equity allocation; >1 increases growth tilt.")
    st.caption("No hedge sleeve or debt instruments are included. NTPC, if selected by its sector return ranking, is treated like any other equity.")

    profile_map = {
        "Conservative": {"equity_share": 0.65, "return_w": 0.32, "quality_w": 0.22, "risk_w": 0.46},
        "Moderate":     {"equity_share": 0.80, "return_w": 0.45, "quality_w": 0.20, "risk_w": 0.35},
        "Aggressive":   {"equity_share": 0.92, "return_w": 0.60, "quality_w": 0.15, "risk_w": 0.25},
    }

    if investor_type == "Custom":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            custom_equity = st.slider("Custom equity share (%)", 25, 95, 80, 1, key="uni_custom_eq") / 100
        with c2:
            custom_return_w = st.slider("Return weight", 0.0, 1.0, 0.45, 0.05, key="uni_custom_ret")
        with c3:
            custom_quality_w = st.slider("Quality weight", 0.0, 1.0, 0.20, 0.05, key="uni_custom_qual")
        with c4:
            custom_risk_w = st.slider("Risk weight", 0.0, 1.0, 0.35, 0.05, key="uni_custom_risk")
        cfg = {"equity_share": custom_equity, "return_w": custom_return_w,
               "quality_w": custom_quality_w, "risk_w": custom_risk_w}
    else:
        cfg = profile_map[investor_type]

    capital = goal_start * 1e7
    target_equity_share = 1.0
    equity_budget = capital
    hedge_budget = 0.0

    # ---------------- Allocation loop ---------------- #
    # Historical two-month adjusted-price return is the ONLY portfolio return input.
    allocation_rows = []
    alloc_errors = []

    for _, row in df.iterrows():
        try:
            ticker = row["Ticker"]
            current_price = _safe_float(row.get("Latest Close Price"))
            if current_price is None or current_price <= 0:
                current_price = _safe_float(row.get("Current Price"))
            if current_price is None or current_price <= 0:
                current_price = _safe_float(row.get("Latest Adjusted Price"))
            # Keep each portfolio row even when price/return data is missing.
            if current_price is None or current_price <= 0:
                current_price = np.nan

            return_pct = _safe_float(row.get("2-Month Return %"))
            historical_return_2m = return_pct / 100.0 if return_pct is not None else None
            # Annualized equivalent is context only and is never the expected-return input.
            if historical_return_2m is None:
                annualized_equivalent = np.nan
                goal_alignment = np.nan
            else:
                annualized_equivalent = ((1.0 + historical_return_2m) ** 6 - 1.0
                                         if historical_return_2m > -1.0 else -1.0)
                goal_alignment = (np.clip(annualized_equivalent / required_cagr, 0, 1.5)
                                  if required_cagr > 0 else np.clip(annualized_equivalent, 0, 1.5))

            try:
                fi = load_fundamentals(ticker) or {}
            except Exception:
                fi = {}

            # ---- Risk: annualized standard deviation of daily returns ----
            try:
                h, _ = load_history(ticker, "1y", "1d", False)
            except Exception:
                h = pd.DataFrame()
            risk_close = None
            if h is not None and not h.empty:
                risk_close = h["Adj Close"] if "Adj Close" in h.columns else h.get("Close")
            std_dev_ann = annualized_std(risk_close)

            # ---- Quality metrics (descriptive only; no DCF value used) ----
            roce = _safe_float(fi.get("returnOnCapitalEmployed"))
            roe = _safe_float(fi.get("returnOnEquity"))
            op_margin = _safe_float(fi.get("operatingMargin"))
            net_margin = _safe_float(fi.get("profitMargin"))
            de = _safe_float(fi.get("debtToEquity"))
            de = de / 100.0 if de is not None else None
            peg = _safe_float(fi.get("pegRatio"))

            quality_score = (
                0.30 * np.clip((roce or 0) / 0.25, 0, 1)
                + 0.25 * np.clip((roe or 0) / 0.35, 0, 1)
                + 0.15 * np.clip((op_margin or 0) / 0.20, 0, 1)
                + 0.10 * np.clip((net_margin or 0) / 0.15, 0, 1)
                + 0.10 * (1 - np.clip((de or 0) / 2.0, 0, 1))
                + 0.10 * (1 - np.clip((peg or 1) / 3.0, 0, 1))
            )

            # Risk score is price-volatility/leverage based; DCF model dispersion
            # is deliberately excluded from the portfolio allocation engine.
            volatility_score = np.clip((std_dev_ann or 0) / 0.60, 0, 1)
            leverage_score = np.clip((de or 0) / 2.0, 0, 1)
            risk_score = float(np.clip(0.70 * volatility_score + 0.30 * leverage_score, 0, 1))

            if risk_score <= 0.25:
                risk_rating = "Low"
            elif risk_score <= 0.50:
                risk_rating = "Moderate"
            elif risk_score <= 0.75:
                risk_rating = "High"
            else:
                risk_rating = "Very High"

            allocation_rows.append({
                "Asset": row["Stock"], "Asset Type": "Equity", "Sector": row["Sector"],
                "Current Price": current_price,
                "Std Dev (Ann %)": std_dev_ann * 100 if std_dev_ann is not None else None,
                "2-Month Return %": historical_return_2m * 100.0 if historical_return_2m is not None else np.nan,
                "Annualized Equivalent %": annualized_equivalent * 100.0 if _safe_float(annualized_equivalent) is not None else np.nan,
                "Goal Alignment": goal_alignment if _safe_float(goal_alignment) is not None else np.nan,
                "Expected Return (₹)": np.nan,
                "Quality Score": float(quality_score),
                "Risk Score": risk_score, "Risk Rating": risk_rating,
                "Notes": "Uses the trailing two-month adjusted-price return; historical performance is not a forecast.",
            })
        except Exception as e:
            alloc_errors.append(f"{row.get('Stock','?')}: {type(e).__name__}: {e}")

    if alloc_errors:
        with st.expander("⚠️ Allocation errors", expanded=False):
            for msg in alloc_errors:
                st.write(msg)

    if not allocation_rows:
        st.error("Allocation could not be built — no valid rows.")
        return

    # ---------------- Allocation and historical-return calculations ---------------- #
    df_alloc = pd.DataFrame(allocation_rows)
    if df_alloc.empty:
        st.error("Allocation could not be built — no selected stocks have valid two-month return and price data.")
        return

    # Default ratios are tilted toward higher trailing returns, with a floor so no stock hits zero.
    st.markdown("### Allocation method")
    m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1])
    with m1:
        alloc_method = st.radio("Default weights", ["Return-tilted (recommended)", "Equal weight"],
                                key="uni_alloc_method", horizontal=True)
    with m2:
        floor_pct = st.slider("Minimum weight per stock (%)", 1.0, 6.0, 3.0, 0.5, key="uni_alloc_floor",
                              help="Guaranteed minimum so no holding is ever cut to zero.")
    with m3:
        tilt_strength = st.slider("Return tilt strength", 0.0, 2.0, 1.0, 0.1, key="uni_alloc_tilt",
                                  help="0 = equal weight; higher = more capital to recent winners.")
    with m4:
        cap_pct = st.slider("Maximum weight per stock (%)", 8.0, 25.0, 15.0, 1.0, key="uni_alloc_cap",
                            help="Limits concentration in any single stock.")

    n_alloc = len(df_alloc)
    if alloc_method.startswith("Return"):
        default_w = compute_return_tilted_weights(df_alloc["2-Month Return %"].values,
                                                  floor=floor_pct / 100.0, tilt=tilt_strength, cap=cap_pct / 100.0)
    else:
        default_w = np.full(n_alloc, 1.0 / n_alloc)
    df_alloc["Allocation Ratio"] = default_w * 100.0
    # Reset any manual grid edits whenever the default weights change.
    ratio_sig = hashlib.md5(",".join(f"{x:.3f}" for x in default_w).encode()).hexdigest()[:10]

    st.markdown("### Edit company allocation ratios")
    st.caption("Defaults follow the method above (percent of capital). You can override any ratio; ratios are normalized to 100% automatically. Expected-return amounts use the trailing two-month return shown here.")
    try:
        editable = st.data_editor(
            df_alloc[["Asset", "Sector", "2-Month Return %", "Allocation Ratio"]],
            column_config={
                "Allocation Ratio": st.column_config.NumberColumn("Allocation Ratio", min_value=0.0, max_value=100.0, step=0.25, format="%.2f"),
                # Values are percentage points (e.g. 7.25), not decimal fractions.
                "2-Month Return %": st.column_config.NumberColumn("2-Month Return (%)", format="%.2f%%", disabled=True),
            },
            disabled=["Asset", "Sector", "2-Month Return %"],
            hide_index=True, use_container_width=True, key=f"momentum_allocation_ratios_{ratio_sig}")
    except Exception as exc:
        st.warning(f"The editable allocation-ratio grid is unavailable ({type(exc).__name__}: {exc}); equal ratios will be used so the portfolio output can still render.")
        editable = df_alloc[["Asset", "Allocation Ratio"]].copy()

    df_alloc = df_alloc.drop(columns=["Allocation Ratio"]).merge(
        editable[["Asset", "Allocation Ratio"]], on="Asset", how="left", validate="one_to_one")
    df_alloc["Allocation Ratio"] = pd.to_numeric(df_alloc["Allocation Ratio"], errors="coerce").fillna(0).clip(lower=0)
    ratio_sum = float(df_alloc["Allocation Ratio"].sum())
    if ratio_sum <= 0:
        df_alloc["Allocation Ratio"] = 1.0
        ratio_sum = float(len(df_alloc))
    df_alloc["Allocation %"] = df_alloc["Allocation Ratio"] / ratio_sum
    df_alloc["Allocation (₹)"] = df_alloc["Allocation %"] * capital
    df_alloc["Units"] = df_alloc.apply(
        lambda x: x["Allocation (₹)"] / x["Current Price"]
        if _safe_float(x["Current Price"]) and _safe_float(x["Current Price"]) > 0 else np.nan, axis=1)
    df_alloc["Allocation (₹ Cr)"] = df_alloc["Allocation (₹)"] / 1e7
    df_alloc["Expected Return (₹)"] = df_alloc["Allocation (₹)"] * (df_alloc["2-Month Return %"] / 100.0)
    df_alloc["Illustrative Value after 2 Months (₹)"] = df_alloc["Allocation (₹)"] + df_alloc["Expected Return (₹)"]
    df_alloc["Asset Type"] = "Equity"
    df_alloc = df_alloc.sort_values("Allocation (₹)", ascending=False).reset_index(drop=True)

    equity_alloc = float(df_alloc["Allocation (₹)"].sum())
    return_coverage_count = int(df_alloc["2-Month Return %"].notna().sum())
    return_coverage_total = int(len(df_alloc))
    full_return_coverage = return_coverage_count == return_coverage_total and return_coverage_total > 0
    if full_return_coverage:
        portfolio_expected_gain = float(df_alloc["Expected Return (₹)"].sum())
        portfolio_return_2m = portfolio_expected_gain / capital if capital > 0 else 0.0
        annualized_equivalent = ((1.0 + portfolio_return_2m) ** 6 - 1.0
                                 if portfolio_return_2m > -1.0 else -1.0)
        illustrative_portfolio_value = capital + portfolio_expected_gain
    else:
        # A partial sum cannot be presented as the return on the entire portfolio.
        portfolio_expected_gain = np.nan
        portfolio_return_2m = np.nan
        annualized_equivalent = np.nan
        illustrative_portfolio_value = np.nan
    weighted_std_dev = float(np.average(
        df_alloc["Std Dev (Ann %)"].fillna(0.0), weights=df_alloc["Allocation (₹)"].clip(lower=0)))
    weighted_risk_score = float(np.average(
        df_alloc["Risk Score"].fillna(0.0), weights=df_alloc["Allocation (₹)"].clip(lower=0)))
    std_norm = float(np.clip(weighted_std_dev / 60.0, 0, 1))
    portfolio_risk_index = 0.70 * weighted_risk_score + 0.30 * std_norm
    if portfolio_risk_index <= 0.25:
        portfolio_risk_rating = "Low"
    elif portfolio_risk_index <= 0.50:
        portfolio_risk_rating = "Moderate"
    elif portfolio_risk_index <= 0.75:
        portfolio_risk_rating = "High"
    else:
        portfolio_risk_rating = "Very High"

    # KPIs make both the return rate and rupee impact visible.
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Goal start", f"₹{goal_start:.2f} Cr")
    c2.metric("Goal target", f"₹{goal_end:.2f} Cr")
    c3.metric("Total allocated", f"₹{equity_alloc / 1e7:.2f} Cr")
    c4.metric("Historical-return gain (2M)", f"₹{portfolio_expected_gain:,.0f}" if full_return_coverage else "N/A")
    c5.metric("Weighted 2-month return", f"{portfolio_return_2m * 100:.2f}%" if full_return_coverage else "N/A")
    c6.metric("Risk rating", portfolio_risk_rating)

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Weighted Std Dev (Ann %)", f"{weighted_std_dev:.1f}%")
    c8.metric("Annualized equivalent*", f"{annualized_equivalent * 100:.1f}%" if full_return_coverage else "N/A")
    c9.metric("Illustrative value after 2 months", f"₹{illustrative_portfolio_value / 1e7:.2f} Cr" if full_return_coverage else "N/A")
    c10.metric("Weighted Risk Score", f"{weighted_risk_score:.3f}")
    st.caption(f"Historical-return coverage: {return_coverage_count}/{return_coverage_total} selected holdings.")
    if not full_return_coverage:
        st.warning("At least one selected holding has no usable two-month return. Per-stock values and allocation charts are still shown, but whole-portfolio gain/return is N/A until price history is complete.")
    st.caption("*Compounding the latest two-month return across six periods is shown for comparison only; it is not a forecast or a reliable long-term projection.")

    # All portfolio charts use the same final allocation dataframe.
    display_df = df_alloc.copy()
    chart_df = display_df.copy()
    chart_df["Std Dev (Ann %)"] = chart_df["Std Dev (Ann %)"].fillna(0.0)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_alloc = px.bar(
            chart_df, x="Asset", y="Allocation (₹ Cr)", color="Sector",
            hover_data={"Allocation %": ":.1%", "Allocation (₹)": ":,.0f",
                        "Current Price": ":.2f", "Allocation (₹ Cr)": ":.3f"},
            title="Portfolio Allocation by Stock (₹ Cr)", template="plotly_dark")
        fig_alloc.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=460,
                                xaxis_tickangle=-45, yaxis_title="Allocated capital (₹ Cr)")
        st.plotly_chart(fig_alloc, use_container_width=True, config=PLOTLY_CFG)

    with chart_right:
        fig_rr = px.scatter(
            chart_df, x="Std Dev (Ann %)", y="2-Month Return %",
            size="Allocation (₹ Cr)", color="Sector", hover_name="Asset",
            hover_data={"Allocation %": ":.1%", "Expected Return (₹)": ":,.0f",
                        "Std Dev (Ann %)": ":.2f", "2-Month Return %": ":.2f"},
            title="Risk vs Historical 2-Month Return", template="plotly_dark")
        fig_rr.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=460,
                             xaxis_title="Annualized volatility (%)", yaxis_title="Trailing 2-month return (%)")
        st.plotly_chart(fig_rr, use_container_width=True, config=PLOTLY_CFG)

    fig_pie = px.pie(display_df, names="Asset", values="Allocation (₹ Cr)",
                     title="Portfolio Allocation Share", hole=0.35, template="plotly_dark")
    fig_pie.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=520)
    st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)

    fig_returns = px.bar(
        display_df.sort_values("2-Month Return %", ascending=False),
        x="Asset", y="2-Month Return %", color="Sector", hover_name="Asset",
        hover_data={"Expected Return (₹)": ":,.0f", "Allocation %": ":.1%"},
        title="Expected Return Basis — Actual Trailing 2-Month Stock Return",
        template="plotly_dark")
    fig_returns.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=430,
                              xaxis_tickangle=-45, yaxis_title="Historical 2-month return (%)")
    st.plotly_chart(fig_returns, use_container_width=True, config=PLOTLY_CFG)

    # Allocation table includes the amount invested and the return/gain it implies.
    st.markdown("#### Portfolio allocation and expected-return table")
    cols_to_show = [
        "Asset", "Sector", "Current Price", "Allocation Ratio", "Allocation %",
        "Allocation (₹)", "Allocation (₹ Cr)", "Units", "2-Month Return %",
        "Expected Return (₹)", "Illustrative Value after 2 Months (₹)",
        "Annualized Equivalent %", "Std Dev (Ann %)", "Risk Score", "Risk Rating",
        "Quality Score", "Goal Alignment", "Notes",
    ]
    present = [c for c in cols_to_show if c in display_df.columns]
    fmt_alloc = {
        "Current Price": "₹{:,.2f}",
        "Allocation Ratio": "{:.2f}",
        "Allocation %": "{:.2%}",
        "Allocation (₹)": "₹{:,.0f}",
        "Allocation (₹ Cr)": "₹{:.3f} Cr",
        "Units": "{:,.2f}",
        "2-Month Return %": "{:+.2f}%",
        "Expected Return (₹)": "₹{:+,.0f}",
        "Illustrative Value after 2 Months (₹)": "₹{:,.0f}",
        "Annualized Equivalent %": "{:+.1f}%",
        "Std Dev (Ann %)": "{:.1f}%",
        "Risk Score": "{:.3f}",
        "Quality Score": "{:.3f}",
        "Goal Alignment": "{:.2f}",
    }
    try:
        st.dataframe(display_df[present].style.format(fmt_alloc, na_rep="N/A"),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(display_df[present], use_container_width=True, hide_index=True)

    # Sector gains are only reported when every stock in that sector has a valid return.
    sector_records = []
    for sector_name, sector_rows in display_df.groupby("Sector", dropna=False):
        allocated = float(sector_rows["Allocation (₹)"].sum())
        sector_has_complete_returns = bool(sector_rows["2-Month Return %"].notna().all())
        gain = float(sector_rows["Expected Return (₹)"].sum()) if sector_has_complete_returns else np.nan
        weighted_ret = (gain / allocated * 100.0) if sector_has_complete_returns and allocated > 0 else np.nan
        sector_records.append({
            "Sector": sector_name,
            "Allocated Capital (₹)": allocated,
            "Historical Return Gain (₹)": gain,
            "Weighted 2-Month Return %": weighted_ret,
        })
    sector_summary = pd.DataFrame(sector_records)
    st.markdown("#### Sector allocation and historical return summary")
    try:
        st.dataframe(sector_summary.style.format({
            "Allocated Capital (₹)": "₹{:,.0f}",
            "Historical Return Gain (₹)": "₹{:+,.0f}",
            "Weighted 2-Month Return %": "{:+.2f}%",
        }, na_rep="N/A"), use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(sector_summary, use_container_width=True, hide_index=True)

    # ---------------- Why this allocation: news + economy rationale ---------------- #
    st.markdown("#### Why this allocation — news & economy rationale")
    st.caption(f"Weights are driven by trailing two-month returns (floor {floor_pct:.1f}%, cap {cap_pct:.0f}%, tilt {tilt_strength:.1f}); "
               f"the news and macro context below (snapshot as of {MACRO_ASOF}) explains why that momentum is plausible and where it could break. "
               "It is context, not a forecast or investment advice.")
    with st.expander("Macro backdrop (RBI, oil, rupee, FPI flows)", expanded=True):
        for line in MACRO_BACKDROP:
            st.markdown(f"- {line}")

    for sector_name in display_df["Sector"].dropna().unique():
        thesis = SECTOR_THESIS.get(sector_name)
        if not thesis:
            continue
        sector_w = float(display_df.loc[display_df["Sector"] == sector_name, "Allocation %"].sum())
        with st.expander(f"{sector_name} — {sector_w * 100:.1f}% of portfolio", expanded=False):
            st.markdown(f"**News:** {thesis['news']}")
            st.markdown(f"**Economy:** {thesis['economy']}")
            st.markdown(f"**Risk to the tilt:** {thesis['risk']}")

    try:
        rationale_df = build_allocation_rationale(display_df, len(display_df)).sort_values("Weight %", ascending=False)
        st.dataframe(rationale_df.style.format({"Weight %": "{:.1f}%"}), use_container_width=True, hide_index=True,
                     column_config={"Rationale": st.column_config.TextColumn("Rationale", width="large")})
    except Exception as exc:
        st.info(f"Per-stock rationale unavailable: {type(exc).__name__}: {exc}")

    # Risk breakdown
    with st.expander("📎 Risk breakdown (volatility · leverage)"):
        st.caption("Risk score = 0.70 × volatility score + 0.30 × leverage score. DCF/model dispersion is not included.")
        eq = df_alloc[df_alloc["Asset Type"].isin(["Equity", "Hedge Equity"])]
        if len(eq):
            risk_breakdown = eq[[
                "Asset", "Sector", "Std Dev (Ann %)", "Risk Score", "Risk Rating",
            ]].copy().sort_values("Std Dev (Ann %)", ascending=False, na_position="last")
            try:
                st.dataframe(
                    risk_breakdown.style.format({
                        "Std Dev (Ann %)": "{:.1f}%",
                        "Risk Score": "{:.3f}",
                    }, na_rep="N/A"),
                    use_container_width=True, hide_index=True,
                )
            except Exception:
                st.dataframe(risk_breakdown, use_container_width=True, hide_index=True)


def render_valuation(ticker, fi, sector_tickers):
    st.subheader("💵 Valuation Models & Comparison (Individual Stock)")
    st.caption("Detailed DCF, DDM and relative valuation for the selected stock.")

    income, balance, cashflow = load_financial_statements(ticker)
    cf_data = compute_free_cash_flow(ticker, income, balance, cashflow)
    fcff = cf_data.get("fcff")
    fcfe = cf_data.get("fcfe")

    current_price = _safe_float(fi.get("currentPrice"))
    eps = _safe_float(fi.get("trailingEps"))
    book_value = _safe_float(fi.get("bookValue"))
    pe = _safe_float(fi.get("trailingPE"))
    pb = _safe_float(fi.get("priceToBook"))
    peg = _safe_float(fi.get("pegRatio"))
    sector_pe = _safe_float(fi.get("sectorPE"))

    avg_pe, avg_pb, avg_peg, avg_ev_ebitda = compute_sector_average(sector_tickers)

    st.markdown("### Macroeconomic Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        rf = st.number_input("Risk-free rate (%)", 0.0, 20.0, DEFAULT_RF, 0.1, key="val_rf") / 100
    with col2:
        inflation = st.number_input("Inflation (%)", 0.0, 20.0, 5.0, 0.1, key="val_inf") / 100
    with col3:
        gdp_growth = st.number_input("GDP growth (%)", 0.0, 20.0, 6.5, 0.1, key="val_gdp") / 100
    st.divider()

    st.markdown("### 📈 Dividend Discount Model (DDM)")
    div_yield = _safe_float(fi.get("dividendYield"))
    payout_ratio = _safe_float(fi.get("payoutRatio"))
    if div_yield is not None and current_price is not None:
        d0 = div_yield * current_price
    elif payout_ratio is not None and eps is not None:
        d0 = payout_ratio * eps
    else:
        d0 = None
    if d0 is None:
        st.info("Dividend data not available. Enter manually.")
        d0 = st.number_input("Manual Dividend per Share (₹)", 0.0, value=5.0, step=0.5)

    ddm_value = None
    if d0 is not None and d0 > 0:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            growth_ddm = st.number_input("Dividend growth rate (%)", 0.0, 20.0, 5.0, 0.1, key="ddm_g") / 100
        with col_d2:
            required_return = st.number_input("Required return / cost of equity (%)", 0.0, 30.0,
                                               rf*100 + 5.0, 0.1, key="ddm_r") / 100
        if required_return > growth_ddm:
            ddm_value = d0 * (1 + growth_ddm) / (required_return - growth_ddm)
            st.metric("Intrinsic Value (DDM)", f"₹{ddm_value:.2f}",
                      delta=f"{((ddm_value/current_price)-1)*100:.1f}%" if current_price else None)
        else:
            st.warning("Required return must be greater than growth rate.")
    st.divider()

    st.markdown("### 💰 Discounted Cash Flow (DCF)")
    if fcff is None and fcfe is None:
        st.warning("Could not compute FCF. Enter manually.")
        fcff_manual = st.number_input("Manual FCFF (₹ Cr)", 0.0, value=1000.0, step=100.0)
        fcfe_manual = st.number_input("Manual FCFE (₹ Cr)", 0.0, value=800.0, step=100.0)
        fcff = fcff_manual * 1e7
        fcfe = fcfe_manual * 1e7
        if fcff == 0 and fcfe == 0:
            fcff = fcfe = None

    fcff_value = None
    fcfe_value = None

    if fcff is not None or fcfe is not None:
        st.subheader("DCF Assumptions")
        col_wacc, col_g, col_shares = st.columns(3)
        with col_wacc:
            wacc = st.number_input("WACC (%)", 0.0, 30.0, rf*100 + 4.0, 0.1, key="dcf_wacc") / 100
        with col_g:
            growth_terminal = st.number_input("Terminal growth rate (%)", 0.0, 10.0, 3.0, 0.1, key="dcf_g") / 100
        with col_shares:
            shares = st.number_input("Shares outstanding (million)", 0.0, value=100.0, step=1.0, key="dcf_shares")
            if shares == 0.0:
                shares_raw = fi.get("sharesOutstanding")
                shares = shares_raw / 1e6 if shares_raw else 100.0

        growth_fcf = st.slider("FCF growth rate (next 5 years, %)", 0.0, 20.0, 5.0, 0.1, key="fcf_g") / 100

        if fcff is not None:
            st.subheader("FCFF Valuation")
            fcff_proj = [fcff * (1 + growth_fcf) ** i for i in range(1, 6)]
            tv_fcff = (fcff_proj[-1] * (1 + growth_terminal) / (wacc - growth_terminal)
                       if wacc > growth_terminal else 0)
            if tv_fcff == 0:
                st.warning("WACC must exceed terminal growth.")
            fcff_pv = sum(f / (1 + wacc) ** i for i, f in enumerate(fcff_proj, start=1))
            if tv_fcff > 0:
                fcff_pv += tv_fcff / (1 + wacc) ** 5
            total_debt = get_item_from_df(balance, ["Total Debt", "Long Term Debt", "Short Term Debt"])
            cash_equiv = get_item_from_df(balance, ["Cash And Cash Equivalents", "Cash"])
            equity_value = (fcff_pv - total_debt + cash_equiv
                            if (total_debt is not None and cash_equiv is not None) else fcff_pv)
            fcff_value = equity_value / (shares * 1e6)
            st.metric("Equity Value per Share (FCFF)", f"₹{fcff_value:.2f}",
                      delta=f"{((fcff_value/current_price)-1)*100:.1f}%" if current_price else None)

            st.subheader("FCFF Sensitivity: Value vs WACC and Terminal Growth")
            wacc_range = np.arange(max(0.01, wacc - 0.02), wacc + 0.03, 0.01)
            growth_range = np.arange(max(0.001, growth_terminal - 0.02),
                                      min(0.10, growth_terminal + 0.03), 0.01)
            sens_data = []
            for w in wacc_range:
                row_vals = []
                for g in growth_range:
                    if w > g:
                        tv = fcff_proj[-1] * (1 + g) / (w - g)
                        pv = sum([f / (1 + w) ** (i+1) for i, f in enumerate(fcff_proj)]) + tv / (1 + w) ** 5
                        if total_debt is not None and cash_equiv is not None:
                            pv = pv - total_debt + cash_equiv
                        row_vals.append(pv / (shares * 1e6))
                    else:
                        row_vals.append(np.nan)
                sens_data.append(row_vals)
            sens_df = pd.DataFrame(sens_data,
                                    index=[f"{w*100:.1f}%" for w in wacc_range],
                                    columns=[f"{g*100:.1f}%" for g in growth_range])
            st.dataframe(sens_df.style.format("{:.2f}").highlight_max(color=GREEN),
                         use_container_width=True)

        if fcfe is not None:
            st.subheader("FCFE Valuation")
            cost_equity = st.number_input("Cost of equity (%)", 0.0, 30.0, rf*100 + 5.0, 0.1, key="fcf_cost_eq") / 100
            fcfe_proj = [fcfe * (1 + growth_fcf) ** i for i in range(1, 6)]
            tv_fcfe = (fcfe_proj[-1] * (1 + growth_terminal) / (cost_equity - growth_terminal)
                       if cost_equity > growth_terminal else 0)
            if tv_fcfe == 0:
                st.warning("Cost of equity must exceed terminal growth.")
            fcfe_pv = sum(f / (1 + cost_equity) ** i for i, f in enumerate(fcfe_proj, start=1))
            if tv_fcfe > 0:
                fcfe_pv += tv_fcfe / (1 + cost_equity) ** 5
            fcfe_value = fcfe_pv / (shares * 1e6)
            st.metric("Equity Value per Share (FCFE)", f"₹{fcfe_value:.2f}",
                      delta=f"{((fcfe_value/current_price)-1)*100:.1f}%" if current_price else None)
    st.divider()

    st.markdown("### ⚖️ Relative Valuation (vs. Sector Average)")
    implied_prices = {}
    if eps and avg_pe: implied_prices["P/E (Sector Avg)"] = eps * avg_pe
    if book_value and avg_pb: implied_prices["P/B (Sector Avg)"] = book_value * avg_pb
    if avg_ev_ebitda and fi.get("ebitda"):
        ev_implied = fi.get("ebitda") * avg_ev_ebitda
        debt = get_item_from_df(balance, ["Total Debt", "Long Term Debt", "Short Term Debt"])
        cash = get_item_from_df(balance, ["Cash And Cash Equivalents", "Cash"])
        if debt is not None and cash is not None:
            shares_out = fi.get("sharesOutstanding")
            if shares_out:
                implied_prices["EV/EBITDA (Sector Avg)"] = (ev_implied - debt + cash) / shares_out
    if sector_pe and eps: implied_prices["P/E (Yahoo Sector)"] = eps * sector_pe

    comp_data = {
        "Metric": ["Current Price", "DDM", "FCFF", "FCFE"] + list(implied_prices.keys()),
        "Value (₹)": [current_price, ddm_value, fcff_value, fcfe_value] + list(implied_prices.values()),
    }
    df_comp = pd.DataFrame(comp_data)
    df_comp["Upside %"] = df_comp["Value (₹)"].apply(
        lambda x: f"{((x/current_price)-1)*100:.1f}%" if x and current_price else "N/A")
    st.dataframe(df_comp.style.format({"Value (₹)": "{:.2f}"}, na_rep="N/A"),
                 hide_index=True, use_container_width=True)

    st.subheader("Sector Averages (from peer group)")
    sector_df = pd.DataFrame({
        "Metric": ["P/E", "P/B", "PEG", "EV/EBITDA"],
        "Sector Avg": [avg_pe, avg_pb, avg_peg, avg_ev_ebitda],
        "Stock": [pe, pb, peg,
                  (fi.get("enterpriseValue") / fi.get("ebitda"))
                  if (fi.get("enterpriseValue") and fi.get("ebitda")) else None],
    })
    st.dataframe(sector_df.style.format({"Sector Avg": "{:.2f}", "Stock": "{:.2f}"}, na_rep="N/A"),
                 hide_index=True, use_container_width=True)

    # Standalone risk snapshot
    st.subheader("Standalone risk — annualized standard deviation")
    h_sel, _ = load_history(ticker, "1y", "1d", False)
    std_sel = annualized_std(h_sel["Close"]) if not h_sel.empty else None
    if std_sel is not None:
        rc1, rc2 = st.columns(2)
        rc1.metric("Std Dev (Annualized)", f"{std_sel * 100:.1f}%")
        rc2.metric("Daily Std Dev", f"{(std_sel / np.sqrt(252)) * 100:.2f}%")
    else:
        st.info("Insufficient price history to compute standard deviation.")

    with st.expander("Raw Financial Data"):
        st.write("Income Statement", income)
        st.write("Balance Sheet", balance)
        st.write("Cash Flow", cashflow)
        st.write("Computed Cash Flow Metrics", cf_data)


# ====================================================================== #
# 9. HEADER HELPERS
# ====================================================================== #
def render_market_strip(fi, period, sector):
    price = _safe_float(fi.get("currentPrice"))
    pe = _safe_float(fi.get("trailingPE"))
    roe = _safe_float(fi.get("returnOnEquity"))
    roce = _safe_float(fi.get("returnOnCapitalEmployed"))
    cap = _safe_float(fi.get("marketCap"))
    cap_text = _fmt_num(cap, prefix="₹", decimals=1)
    cards = [
        ("Last price", _fmt_num(price, prefix="₹", decimals=2), "Live / cached equity price"),
        ("Market cap", cap_text, sector + " universe"),
        ("P / E", _fmt_ratio(pe), "Trailing valuation multiple"),
        ("ROE / ROCE", f"{_fmt_pct(roe)} / {_fmt_pct(roce)}", f"Selected timeframe · {period.upper()}"),
    ]
    html = ['<div class="ticker-strip">']
    for label, value, meta in cards:
        html.append(f'<div class="ticker-card"><div class="ticker-label">{label}</div>'
                    f'<div class="ticker-value">{value}</div><div class="ticker-meta">{meta}</div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ====================================================================== #
# 10. MAIN
# ====================================================================== #
def main():
    inject_css()

    with st.sidebar:
        render_brand()
        st.markdown('<div class="small-note">INSTITUTIONAL RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("**Research Universe**")
        sector = st.selectbox("Sector", list(SECTORS.keys()), index=0, label_visibility="collapsed")
        sector_tickers = SECTORS[sector]
        ticker_names = [n for n, _ in sector_tickers]
        selected_name = st.selectbox("Stock", ticker_names, index=0, label_visibility="collapsed")
        selected_ticker = dict(sector_tickers)[selected_name]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Chart Controls**")
        interval = st.selectbox("Candle interval", ["1d", "1h", "15m"], index=0,
                                format_func=lambda x: {"1d": "Daily", "1h": "1 Hour", "15m": "15 Minutes"}[x])
        period_options = {
            "15m": ["1d", "5d", "1mo"],
            "1h": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            "1d": ["1mo", "6mo", "1y", "5y", "max"],
        }
        period_labels = {
            "1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months",
            "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years", "max": "Max",
        }
        available_periods = period_options[interval]
        default_index = 0 if interval in {"15m", "1h"} else min(len(available_periods) - 1, 2)
        period = st.selectbox("Lookback", available_periods, index=default_index,
                              format_func=lambda x: period_labels[x])
        allow_demo = st.toggle("Allow synthetic fallback", value=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Coverage**")
        st.markdown('<div class="small-note">Indian equities across Pharmaceuticals, IT & Technology and Energy & Power. '
                    'Portfolio returns use trailing two-month adjusted-price performance; risk uses annualized standard deviation of daily returns. '
                    'DCF valuation remains separate and does not feed portfolio expected returns.</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="small-note">Data: Yahoo Finance · Cached analytics · Fallback mode available</div>',
                    unsafe_allow_html=True)

    fi = load_fundamentals(selected_ticker)
    df_hist, is_demo = load_history(selected_ticker, period, interval, allow_demo)

    render_hero(selected_name, selected_ticker, sector, is_demo)
    render_market_strip(fi, period, sector)
    render_index_marquee()
    st.caption("Portfolio selection uses five stocks each from Pharmaceuticals, Energy & Power, and IT & Technology, ranked by trailing two-month adjusted-price return. No hedging instruments are used.")

    if is_demo:
        st.warning("Live data was unavailable — synthetic fallback is being displayed for the selected timeframe.")

    st.markdown(
        '<div class="section-head"><div><h3>Research modules</h3><p>Move from screening to price action, peers, factor risk, DCF valuation and portfolio allocation.</p></div></div>',
        unsafe_allow_html=True,
    )

    tab_screener, tab_charts, tab_compare, tab_factor, tab_val, tab_alloc = st.tabs(
        ["01 · Overview", "02 · Price Action", "03 · Peer Lab",
         "04 · Factor Models", "05 · Valuation", "06 · Portfolio Allocation"]
    )

    with tab_screener:
        render_screener(selected_ticker, fi)

    with tab_charts:
        st.markdown('<div class="section-head"><div><h3>Price action & technical structure</h3>'
                    '<p>Candlesticks · moving averages · VWAP · volume · RSI</p></div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chart-state">◉ {period_labels[period]} lookback · {interval.upper()} candles</div>',
                    unsafe_allow_html=True)
        render_charts(selected_ticker, df_hist, interval=interval, period=period)

    with tab_compare:
        render_compare(sector, sector_tickers)

    with tab_factor:
        render_factor_models(selected_ticker, sector, ALL_STOCKS)

    with tab_val:
        render_valuation(selected_ticker, fi, sector_tickers)

    with tab_alloc:
        render_universe_valuation()


if __name__ == "__main__":
    main()
