import streamlit as st
import pandas as pd
import numpy as np

from core.ui import inject_css, summary_cards, row_card
from core.data import load_daily, safe_last_price
from core.channel import compute_regression_channel, pick_nearest_rails
from core.signals import decide_signal_from_channel
from core.backtest import simulate_tp_sl_first_hit

# -----------------------------
# 기본 유니버스
# -----------------------------
DEFAULT_UNIVERSE = [
    "NVDA","AAPL","MSFT","AMZN","META","GOOGL","TSLA","AVGO","NFLX","ORCL",
    "AMD","INTC","QCOM","MU","SMCI","PLTR","PANW","CRWD","NOW","ADBE",
    "QQQ","SPY","VOO","IWM","TQQQ","SQQQ","SOXL","SOXS",
    "COIN","MSTR","RIOT","MARA",
    "XOM","CVX","JPM","BAC","GS","UNH","JNJ","LLY","PFE",
]

def _parse_universe(text: str) -> list[str]:
    if not text:
        return []
    raw = (
        text.replace("\n", ",")
            .replace(" ", ",")
            .replace("\t", ",")
            .split(",")
    )
    out = []
    for s in raw:
        s = (s or "").strip().upper()
        if s and s not in out:
            out.append(s)
    return out

@st.cache_data(ttl=60 * 20, show_spinner=False)
def analyze_one_symbol(
    symbol: str,
    period: str,
    lookback: int,
    use_log: bool,
    tp_pct: float,
    sl_pct: float,
    horizon: int,
) -> dict:

    df = load_daily(symbol, period=period)
    st.write("DEBUG", symbol, df.shape)

    # 1) 완전 빈 데이터
    if df is None or df.empty:
        return {"symbol": symbol, "error": "데이터 부족(빈 데이터)"}

    # 2) 최소 봉 수 체크
    min_bars = 80
    if len(df) < min_bars:
        return {"symbol": symbol, "error": f"데이터 부족({len(df)}봉 < {min_bars}봉)"}

    # ✅ 현재가 먼저 확보 (중요)
    price = safe_last_price(df)
    if price is None:
        return {"symbol": symbol, "error": "가격 없음"}

    # 3) lookback 자동 조절
    effective_lookback = int(min(lookback, len(df) - 20))
    if effective_lookback < 60:
        effective_lookback = 60

    # 4) 회귀 채널
    ch = compute_regression_channel(
        df,
        lookback=effective_lookback,
        use_log=use_log,
        k_list=[-2, -1, 0, 1, 2],
    )
    if not ch:
        return {"symbol": symbol, "error": "채널 계산 실패"}

    rail_hint = pick_nearest_rails(ch, price)
    sig = decide_signal_from_channel(
        price=price,
        channel=ch,
        rail_hint=rail_hint,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )

    # -----------------------------
    # 간이 과거 샘플링 (1차 버전)
    # -----------------------------
    d = df.copy()
    close = d["Close"].astype(float).values

    recent = close[-effective_lookback:]
    med = float(np.median(recent))

    x = np.arange(len(recent), dtype=float)
    y = np.log(recent) if use_log else recent
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    slope = (((x - x_mean) * (y - y_mean)).sum() / denom) if denom != 0 else 0.0
    slope_pct = (
        float(slope * 100.0)
        if use_log
        else float((slope / (np.mean(recent) + 1e-9)) * 100.0)
    )

    if sig["side"] == "LONG":
        mask = close <= med
        if slope_pct < 0:
            mask = mask & (close <= med * 0.99)
    elif sig["side"] == "SHORT":
        mask = close >= med
        if slope_pct > 0:
            mask = mask & (close >= med * 1.01)
    else:
        mask = np.zeros(len(close), dtype=bool)

    bt = {"n": 0}
    if sig["side"] in ("LONG", "SHORT"):
        bt = simulate_tp_sl_first_hit(
            df=d,
            side=sig["side"],
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            horizon_days=horizon,
            signal_mask=mask,
        )

    out = {
        "symbol": symbol,
        "price": price,
        "side": sig.get("side"),
        "strength": sig.get("strength"),
        "score": sig.get("score"),
        "reason": sig.get("reason"),
        "entry": sig.get("entry"),
        "tp": sig.get("tp"),
        "sl": sig.get("sl"),
        "rr": sig.get("rr"),
        "entry_zone_low": sig.get("entry_zone_low"),
        "entry_zone_high": sig.get("entry_zone_high"),
        "support": sig.get("support"),
        "mid": sig.get("mid"),
        "resist": sig.get("resist"),
        "slope_pct_per_day": sig.get("slope_pct_per_day"),
        "tp_sl_winrate": bt.get("winrate"),
        "tp_sl_n": bt.get("n"),
        "tp_sl_avg_days": bt.get("avg_days"),
        "tp_sl_avg_ret": bt.get("avg_ret"),
        "channel_lookback": effective_lookback,
        "use_log": use_log,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "horizon": horizon,
        "error": None,
    }
    return out

def main():
    st.set_page_config(
        page_title="Angle Lab (회귀채널 + TP/SL 확률)",
        page_icon="📐",
        layout="wide",
    )
    inject_css()

    st.title("📐 Angle Lab")
    st.caption(
        "회귀 채널 기반 빗각 분석 + TP/SL 먼저 도달 확률 스캐너 (1차 패치)"
    )

    with st.expander("⚙️ 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            period = st.selectbox("데이터 기간", ["1y", "2y", "3y", "5y"], index=2)
        with c2:
            lookback = st.number_input(
                "채널 Lookback(일)", min_value=60, max_value=400, value=200, step=10
            )
        with c3:
            use_log = st.checkbox("log 회귀 사용(추천)", value=True)
        with c4:
            horizon = st.number_input(
                "백테스트 관찰 기간(일)", min_value=5, max_value=30, value=10, step=1
            )

        c5, c6, c7 = st.columns(3)
        with c5:
            tp_pct = st.number_input(
                "TP(목표) %", min_value=1.0, max_value=30.0, value=8.0, step=0.5
            )
        with c6:
            sl_pct = st.number_input(
                "SL(손절) %", min_value=0.5, max_value=20.0, value=4.0, step=0.5
            )
        with c7:
            max_items = st.number_input(
                "스캔 최대 종목수", min_value=10, max_value=150, value=40, step=5
            )

        st.markdown("---")
        universe_mode = st.radio(
            "유니버스 선택", ["기본 유니버스", "직접 입력"], horizontal=True
        )
        if universe_mode == "직접 입력":
            universe_text = st.text_area(
                "티커 목록(쉼표/공백/줄바꿈 가능)",
                value=",".join(DEFAULT_UNIVERSE),
            )
            universe = _parse_universe(universe_text)
        else:
            universe = DEFAULT_UNIVERSE[:]

        if len(universe) > int(max_items):
            universe = universe[: int(max_items)]
            st.caption(f"※ 성능을 위해 상위 {max_items}개만 스캔합니다.")

    run = st.button("🚀 스캔 실행", use_container_width=True)

    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None

    if run:
        rows = []
        with st.spinner("스캔 중... (처음은 느릴 수 있음)"):
            for sym in universe:
                rows.append(
                    analyze_one_symbol(
                        symbol=sym,
                        period=period,
                        lookback=int(lookback),
                        use_log=bool(use_log),
                        tp_pct=float(tp_pct),
                        sl_pct=float(sl_pct),
                        horizon=int(horizon),
                    )
                )
        st.session_state["scan_results"] = rows

    results = st.session_state.get("scan_results")
    if not results:
        st.info("스캔 실행을 누르면 결과가 나옵니다.")
        st.stop()

    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    summary_cards({
        "total": len(ok),
        "long": sum(1 for r in ok if r.get("side") == "LONG"),
        "short": sum(1 for r in ok if r.get("side") == "SHORT"),
        "hold": sum(1 for r in ok if r.get("side") == "HOLD"),
    })

    st.subheader("🧾 카드 뷰")
    for r in ok[:40]:
        row_card(r)

    if err:
        with st.expander("⚠️ 실패 목록"):
            for r in err:
                st.write(f"- {r['symbol']}: {r['error']}")

if __name__ == "__main__":
    main()
