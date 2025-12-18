import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_daily(symbol: str, period: str = "3y") -> pd.DataFrame:
    """
    일봉 데이터 로드. auto_adjust=False(원본 OHLC) 유지.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval="1d", auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={c: c.title() for c in df.columns})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def safe_last_price(df: pd.DataFrame) -> float | None:
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
