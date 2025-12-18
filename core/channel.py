import numpy as np
import pandas as pd

def _linreg(x: np.ndarray, y: np.ndarray):
    """
    간단 선형회귀(최소제곱). scipy 없이.
    """
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0, y_mean
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)

def compute_regression_channel(
    df: pd.DataFrame,
    lookback: int = 200,
    use_log: bool = True,
    k_list: list[float] = None
) -> dict:
    """
    회귀 채널 레일 계산.
    - use_log=True면 log(Close) 회귀(장기에도 안정)
    - k_list: 표준편차 배수 레일(예: [-2,-1,0,1,2])
    """
    if k_list is None:
        k_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

    if df is None or df.empty or "Close" not in df.columns:
        return {}

    d = df.tail(max(30, lookback)).copy()
    close = d["Close"].astype(float).values
    n = len(close)
    if n < 30:
        return {}

    x = np.arange(n, dtype=float)
    y = np.log(close) if use_log else close

    slope, intercept = _linreg(x, y)
    y_hat = slope * x + intercept
    resid = y - y_hat
    sigma = float(np.std(resid, ddof=1)) if n >= 3 else float(np.std(resid))

    # 현재(마지막)에서의 중심선 값
    x_now = float(n - 1)
    y_mid_now = slope * x_now + intercept

    # 레일 값(가격)
    rails = {}
    for k in k_list:
        yk = y_mid_now + float(k) * sigma
        rails[k] = float(np.exp(yk) if use_log else yk)

    # 채널 "기울기"를 가격 단위로 대충 변환(해석용)
    # log 회귀면 slope는 log단위/일 -> 대략 일간 %기울기
    slope_pct_per_day = float(slope * 100.0) if use_log else float((slope / (close.mean() + 1e-9)) * 100.0)

    out = {
        "lookback": int(lookback),
        "use_log": bool(use_log),
        "slope": float(slope),
        "slope_pct_per_day": float(slope_pct_per_day),
        "sigma": float(sigma),
        "rails": rails,          # {k: price}
        "mid_now": float(np.exp(y_mid_now) if use_log else y_mid_now),
        "last_close": float(close[-1]),
        "k_list": list(k_list),
    }
    return out

def pick_nearest_rails(channel: dict, price: float) -> dict:
    """
    현재가 주변에서 가까운 지지/저항 레일 선택.
    """
    if not channel or not channel.get("rails"):
        return {}
    rails = channel["rails"]
    ks = sorted(rails.keys())
    # 아래쪽(지지): price 이하 중 가장 큰 rail
    supports = [(k, rails[k]) for k in ks if rails[k] <= price]
    resists  = [(k, rails[k]) for k in ks if rails[k] >= price]
    sup = max(supports, key=lambda t: t[1]) if supports else (min(ks), rails[min(ks)])
    res = min(resists, key=lambda t: t[1]) if resists else (max(ks), rails[max(ks)])
    return {
        "support_k": sup[0], "support": float(sup[1]),
        "resist_k": res[0], "resist": float(res[1]),
        "mid": float(channel.get("mid_now", price)),
    }
