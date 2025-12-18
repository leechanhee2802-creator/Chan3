import pandas as pd
import numpy as np

def simulate_tp_sl_first_hit(
    df: pd.DataFrame,
    side: str,
    tp_pct: float,
    sl_pct: float,
    horizon_days: int = 10,
    signal_mask: np.ndarray | None = None,
) -> dict:
    """
    과거 시그널 발생 지점에서,
    entry=그날 종가 기준으로 TP/SL 먼저 도달했는지 집계.

    LONG:
      TP = entry*(1+tp%)
      SL = entry*(1-sl%)

    SHORT:
      TP = entry*(1-tp%)
      SL = entry*(1+sl%)

    판정은 다음날부터 horizon_days까지 High/Low로 먼저 터진 쪽을 체크.
    """
    if df is None or df.empty:
        return {"n": 0}

    if side not in ("LONG", "SHORT"):
        return {"n": 0}

    d = df.copy()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in d.columns:
            return {"n": 0}

    close = d["Close"].astype(float).values
    high = d["High"].astype(float).values
    low = d["Low"].astype(float).values
    n = len(d)

    if signal_mask is None:
        # 기본: 모든 날을 시그널 후보로 보지 말고, 마지막 horizon 제외
        signal_mask = np.zeros(n, dtype=bool)
        signal_mask[:-horizon_days-1] = True

    idxs = np.where(signal_mask)[0]
    # horizon 고려: 뒤쪽은 제외
    idxs = [i for i in idxs if i + 1 < n and i + horizon_days < n]
    if len(idxs) == 0:
        return {"n": 0}

    wins = 0
    losses = 0
    time_to_hit = []
    ret_list = []

    for i in idxs:
        entry = close[i]
        if entry <= 0:
            continue

        if side == "LONG":
            tp = entry * (1.0 + tp_pct / 100.0)
            sl = entry * (1.0 - sl_pct / 100.0)
        else:
            tp = entry * (1.0 - tp_pct / 100.0)
            sl = entry * (1.0 + sl_pct / 100.0)

        hit = None
        hit_day = None

        for j in range(i + 1, min(n, i + 1 + horizon_days)):
            # 같은 날 TP/SL 동시 터치 가능성 -> 보수적으로 SL 우선(실전 안전)
            if side == "LONG":
                sl_hit = low[j] <= sl
                tp_hit = high[j] >= tp
                if sl_hit and tp_hit:
                    hit = "SL"; hit_day = j; break
                if sl_hit:
                    hit = "SL"; hit_day = j; break
                if tp_hit:
                    hit = "TP"; hit_day = j; break
            else:
                # 숏: TP는 아래 도달(LOW), SL은 위 도달(HIGH)
                tp_hit = low[j] <= tp
                sl_hit = high[j] >= sl
                if sl_hit and tp_hit:
                    hit = "SL"; hit_day = j; break
                if sl_hit:
                    hit = "SL"; hit_day = j; break
                if tp_hit:
                    hit = "TP"; hit_day = j; break

        if hit is None:
            # horizon 내 미도달 -> 중립 처리(확률 계산에서 제외하면 과장될 수 있어)
            # 1차 버전은 "미도달"을 별도로 카운트하고, winrate는 TP/(TP+SL)로 산출
            continue

        if hit == "TP":
            wins += 1
            time_to_hit.append(hit_day - i)
            # 수익률
            if side == "LONG":
                ret_list.append((tp / entry - 1.0) * 100.0)
            else:
                ret_list.append((entry / tp - 1.0) * 100.0)
        else:
            losses += 1
            time_to_hit.append(hit_day - i)
            if side == "LONG":
                ret_list.append((sl / entry - 1.0) * 100.0)
            else:
                ret_list.append((entry / sl - 1.0) * 100.0)

    total = wins + losses
    if total == 0:
        return {"n": 0}

    winrate = wins / total * 100.0
    avg_days = float(np.mean(time_to_hit)) if time_to_hit else None
    avg_ret = float(np.mean(ret_list)) if ret_list else None

    return {
        "n": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": float(winrate),
        "avg_days": float(avg_days) if avg_days is not None else None,
        "avg_ret": float(avg_ret) if avg_ret is not None else None,
    }
