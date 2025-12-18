import numpy as np

def rr_ratio(entry: float, tp: float, sl: float) -> float | None:
    try:
        entry = float(entry); tp = float(tp); sl = float(sl)
    except Exception:
        return None
    if entry <= 0:
        return None
    reward = tp - entry
    risk = entry - sl
    if risk <= 0:
        return None
    return float(reward / risk)

def decide_signal_from_channel(
    price: float,
    channel: dict,
    rail_hint: dict,
    tp_pct: float,
    sl_pct: float,
) -> dict:
    """
    1차 버전 정책:
    - 회귀채널 중심선(mid) 대비 위치 + 기울기(slope_pct_per_day)로 방향 결정
    - 진입(entry): 현재가 기준(단, 레일 기반 "권장 진입 구간"도 같이 제시)
    - TP/SL: entry 기준 %로 산출 (TP/SL 확률 백테스트와 동일 정의)
    - STRONG/WEAK: RR + 레일 여유 + 기울기 기반 스코어
    """
    if not channel or not rail_hint:
        return {"side": "HOLD", "strength": "WEAK", "reason": "채널 계산 실패", "entry": None, "tp": None, "sl": None}

    mid = float(rail_hint.get("mid", price))
    slope_pct = float(channel.get("slope_pct_per_day", 0.0))
    support = float(rail_hint.get("support", price))
    resist  = float(rail_hint.get("resist", price))

    # 방향 점수(간단)
    pos = (price - mid) / (mid + 1e-9)   # 중심선 대비 위치(+면 상단)
    trend = slope_pct                    # +면 우상향

    # 기본 방향:
    # - 우상향(trend>0)이고 mid 아래/근처면 LONG 우선
    # - 우하향(trend<0)이고 mid 위/근처면 SHORT 우선
    side = "HOLD"
    if trend >= 0.0 and pos <= 0.015:   # mid 아래~근처
        side = "LONG"
    elif trend <= 0.0 and pos >= -0.015: # mid 위~근처
        side = "SHORT"
    else:
        # 극단 구간(레일 상/하단) 역추세 시도(약하게)
        # 상단 과열이면 SHORT, 하단 과매도면 LONG
        if price >= resist * 0.995:
            side = "SHORT"
        elif price <= support * 1.005:
            side = "LONG"
        else:
            side = "HOLD"

    entry = float(price)

    if side == "LONG":
        tp = entry * (1.0 + tp_pct / 100.0)
        sl = entry * (1.0 - sl_pct / 100.0)
        # 권장 진입 구간(레일 기반): support~mid
        entry_zone_low = min(support, mid)
        entry_zone_high = max(support, mid)
    elif side == "SHORT":
        tp = entry * (1.0 - tp_pct / 100.0)
        sl = entry * (1.0 + sl_pct / 100.0)
        # 권장 진입 구간(레일 기반): mid~resist
        entry_zone_low = min(mid, resist)
        entry_zone_high = max(mid, resist)
    else:
        tp = None
        sl = None
        entry_zone_low = None
        entry_zone_high = None

    # RR (롱/숏 각각 정의)
    rr = None
    if side == "LONG" and tp is not None and sl is not None:
        rr = rr_ratio(entry, tp, sl)
    if side == "SHORT" and tp is not None and sl is not None:
        # 숏은 수익=entry-tp, 리스크=sl-entry
        reward = entry - tp
        risk = sl - entry
        rr = float(reward / risk) if risk > 0 else None

    # 강도 스코어(0~100)
    # - RR 좋을수록 ↑
    # - 레일까지 여유(롱이면 resist까지 거리, 숏이면 support까지 거리) ↑
    # - 추세 기울기(방향이랑 일치할수록 ↑)
    score = 50.0
    if side != "HOLD":
        if rr is not None:
            score += min(25.0, max(0.0, (rr - 1.0) * 15.0))  # rr=2면 +15
        if side == "LONG":
            room = (resist - price) / (price + 1e-9)
            score += min(15.0, max(0.0, room * 200.0))       # 5% room -> +10
            score += min(10.0, max(-10.0, trend * 1.5))      # trend(+면 가점)
        else:
            room = (price - support) / (price + 1e-9)
            score += min(15.0, max(0.0, room * 200.0))
            score += min(10.0, max(-10.0, (-trend) * 1.5))   # 숏은 trend 음수면 가점
    else:
        score = 40.0

    strength = "STRONG" if score >= 70 else ("WEAK" if score < 55 else "MID")

    # 짧은 사유
    if side == "HOLD":
        reason = "중심선/레일 애매구간(관망)"
    else:
        reason = f"채널기울기 {trend:+.2f}%/day · 중심선대비 {pos*100:+.2f}%"

    return {
        "side": side,
        "strength": strength,
        "score": float(score),
        "reason": reason,
        "entry": float(entry) if entry else None,
        "tp": float(tp) if tp else None,
        "sl": float(sl) if sl else None,
        "entry_zone_low": float(entry_zone_low) if entry_zone_low else None,
        "entry_zone_high": float(entry_zone_high) if entry_zone_high else None,
        "support": float(support),
        "resist": float(resist),
        "mid": float(mid),
        "slope_pct_per_day": float(channel.get("slope_pct_per_day", 0.0)),
        "rr": float(rr) if rr is not None else None,
    }
