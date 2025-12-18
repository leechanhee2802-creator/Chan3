import pandas as pd
import yfinance as yf
import time

def load_daily(symbol: str, period: str = "2y") -> pd.DataFrame:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    # 1️⃣ Yahoo download (history보다 안정적)
    for _ in range(3):
        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                need = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in need):
                    return df[need].dropna()
                return df.dropna()
        except Exception:
            pass
        time.sleep(0.8)

    # 2️⃣ Stooq 백업 (Yahoo 막히면 여기서라도 받음)
    try:
        stooq = f"{symbol}.US".lower()
        url = f"https://stooq.com/q/d/l/?s={stooq}&i=d"
        df2 = pd.read_csv(url)
        if df2 is None or df2.empty:
            return pd.DataFrame()
        df2["Date"] = pd.to_datetime(df2["Date"])
        df2 = df2.set_index("Date").sort_index()
        return df2[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def safe_last_price(df: pd.DataFrame):
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
