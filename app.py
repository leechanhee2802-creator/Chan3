import streamlit as st
import pandas as pd
import numpy as np

from core.ui import inject_css, summary_cards, row_card
from core.data import load_daily, safe_last_price
from core.channel import compute_regression_channel, pick_nearest_rails
from core.signals import decide_signal_from_channel
from core.backtest import simulate_tp_sl_first_hit

# -----------------------------
# 기본 유니버스(1차 버전)
# - 너무 많으면 Streamlit Cloud에서 느려짐
# - 너가 원하는 대로 나중에 S&P500/나스닥100로 확장 가능
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
    # 쉼표/공백/줄바꿈 혼용 허용
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

# 1) 완전 빈 경우
if df is None or df.empty:
    return {"symbol": symbol, "error": "데이터 부족(빈 데이터)"}

# 2) lookback 자동 조절 (기간 짧거나 결측이 있어도 동작)
#    회귀채널은 최소 60봉은 있어야 의미가 있어서 60은 하한으로 둠
min_bars = 80
if len(df) < min_bars:
    return {"symbol": symbol, "error": f"데이터 부족({len(df)}봉 < {min_bars}봉)"}

effective_lookback = int(min(lookback, len(df) - 20))
if effective_lookback < 60:
    effective_lookback = 60

price = safe_last_price(df)
if price is None:
    return {"symbol": symbol, "error": "가격 없음"}

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

    # 시그널 마스크(과거에 "비슷한 조건"일 때만 모아 확률 계산)
    # 1차 버전: 같은 side가 나오는 날만 표본으로 사용(간단하지만 직관적)
    # -> 과최적화 방지 위해 너무 복잡하게 안 함.
    # 만들고 싶으면: "가격이 mid 근처" 같은 조건도 추가 가능.
    d = df.copy()
    close = d["Close"].astype(float).values

    # 과거 시점별로 채널을 매번 계산하면 느려서,
    # 1차 버전은 "단순 트리거"로 샘플을 만들자:
    # LONG이면: 종가가 최근 lookback 중앙값 이하 & 최근 기울기 양수 근사
    # SHORT이면: 종가가 최근 lookback 중앙값 이상 & 최근 기울기 음수 근사
    # (정교한 버전은 2차 패치로)
    recent = close[-lookback:]
    med = float(np.median(recent))
    # 기울기 근사: lookback 구간에서 선형회귀
    x = np.arange(len(recent), dtype=float)
    y = np.log(recent) if use_log else recent
    x_mean = x.mean(); y_mean = y.mean()
    denom = ((x - x_mean)**2).sum()
    slope = (((x - x_mean) * (y - y_mean)).sum() / denom) if denom != 0 else 0.0
    slope_pct = float(slope * 100.0) if use_log else float((slope / (np.mean(recent)+1e-9)) * 100.0)

    if sig["side"] == "LONG":
        mask = (close <= med)  # 간단 조건
        # 추세도 양수인 구간만
        if slope_pct < 0:
            mask = mask & (close <= med * 0.99)
    elif sig["side"] == "SHORT":
        mask = (close >= med)
        if slope_pct > 0:
            mask = mask & (close >= med * 1.01)
    else:
        mask = np.zeros(len(close), dtype=bool)

    bt = {"n": 0}
    if sig["side"] in ("LONG","SHORT"):
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
        "channel_lookback": lookback,
        "use_log": use_log,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "horizon": horizon,
        "error": None,
    }
    return out

def main():
    st.set_page_config(page_title="Angle Lab (회귀채널 + TP/SL 확률)", page_icon="📐", layout="wide")
    inject_css()

    st.title("📐 Angle Lab")
    st.caption("회귀(Regression) 채널로 빗각 레일을 만들고, TP/SL 먼저 도달 확률로 LONG/SHORT 후보를 추립니다. (1차 패치 버전)")

    with st.expander("⚙️ 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            period = st.selectbox("데이터 기간", ["1y","2y","3y","5y"], index=2)
        with c2:
            lookback = st.number_input("채널 Lookback(일)", min_value=60, max_value=400, value=200, step=10)
        with c3:
            use_log = st.checkbox("log 회귀 사용(추천)", value=True)
        with c4:
            horizon = st.number_input("백테스트 관찰 기간(일)", min_value=5, max_value=30, value=10, step=1)

        c5, c6, c7 = st.columns(3)
        with c5:
            tp_pct = st.number_input("TP(목표) %", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
        with c6:
            sl_pct = st.number_input("SL(손절) %", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
        with c7:
            max_items = st.number_input("스캔 최대 종목수", min_value=10, max_value=150, value=40, step=5)

        st.markdown("---")
        universe_mode = st.radio("유니버스 선택", ["기본 유니버스", "직접 입력"], horizontal=True)
        if universe_mode == "직접 입력":
            universe_text = st.text_area("티커 목록(쉼표/공백/줄바꿈 가능)", value=",".join(DEFAULT_UNIVERSE))
            universe = _parse_universe(universe_text)
        else:
            universe = DEFAULT_UNIVERSE[:]

        if len(universe) > int(max_items):
            universe = universe[: int(max_items)]
            st.caption(f"※ 성능을 위해 상위 {max_items}개만 스캔합니다.")

    st.markdown("")

    # 필터/검색
    cF1, cF2, cF3, cF4 = st.columns([2,2,2,2])
    with cF1:
        q = st.text_input("검색(티커)", value="")
    with cF2:
        side_filter = st.selectbox("방향 필터", ["ALL","LONG","SHORT","HOLD"], index=0)
    with cF3:
        strength_filter = st.selectbox("강도 필터", ["ALL","STRONG","MID","WEAK"], index=0)
    with cF4:
        sort_key = st.selectbox("정렬", ["Score", "TP/SL 확률", "샘플수", "RR"], index=1)

    run = st.button("🚀 스캔 실행", use_container_width=True)

    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None

    if run:
        rows = []
        with st.spinner("스캔 중... (처음은 느릴 수 있음)"):
            for sym in universe:
                r = analyze_one_symbol(
                    symbol=sym,
                    period=period,
                    lookback=int(lookback),
                    use_log=bool(use_log),
                    tp_pct=float(tp_pct),
                    sl_pct=float(sl_pct),
                    horizon=int(horizon),
                )
                rows.append(r)
        st.session_state["scan_results"] = rows

    results = st.session_state.get("scan_results")

    if not results:
        st.info("스캔 실행을 누르면 결과가 나옵니다.")
        st.stop()

    # 에러 제외
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    # 통계
    stats = {
        "total": len(ok),
        "long": sum(1 for r in ok if r.get("side") == "LONG"),
        "short": sum(1 for r in ok if r.get("side") == "SHORT"),
        "hold": sum(1 for r in ok if r.get("side") == "HOLD"),
    }
    summary_cards(stats)

    # 필터 적용
    view = ok[:]
    if q.strip():
        qq = q.strip().upper()
        view = [r for r in view if qq in (r.get("symbol","").upper())]

    if side_filter != "ALL":
        view = [r for r in view if r.get("side") == side_filter]

    if strength_filter != "ALL":
        view = [r for r in view if r.get("strength") == strength_filter]

    # 정렬
    def key_score(r): return float(r.get("score") or -1)
    def key_wr(r): return float(r.get("tp_sl_winrate") or -1)
    def key_n(r): return float(r.get("tp_sl_n") or -1)
    def key_rr(r): return float(r.get("rr") or -1)

    if sort_key == "Score":
        view = sorted(view, key=key_score, reverse=True)
    elif sort_key == "TP/SL 확률":
        view = sorted(view, key=key_wr, reverse=True)
    elif sort_key == "샘플수":
        view = sorted(view, key=key_n, reverse=True)
    else:
        view = sorted(view, key=key_rr, reverse=True)

    # 상단 표(한눈에)
    st.subheader("📋 결과 요약 테이블")
    table = []
    for r in view:
        table.append({
            "Symbol": r["symbol"],
            "Side": r["side"],
            "Strength": r["strength"],
            "Score": round(float(r.get("score") or 0), 0),
            "Price": round(float(r.get("price") or 0), 2),
            "TP/SL%": f"{tp_pct:.1f}/{sl_pct:.1f}",
            "WinRate(%)": round(float(r.get("tp_sl_winrate") or 0), 1) if r.get("tp_sl_winrate") is not None else None,
            "N": int(r.get("tp_sl_n") or 0),
            "RR": round(float(r.get("rr") or 0), 2) if r.get("rr") is not None else None,
            "Entry": round(float(r.get("entry") or 0), 2) if r.get("entry") is not None else None,
            "TP": round(float(r.get("tp") or 0), 2) if r.get("tp") is not None else None,
            "SL": round(float(r.get("sl") or 0), 2) if r.get("sl") is not None else None,
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("🧾 카드 뷰")
    st.caption("※ 1차 버전은 속도를 위해 백테스트 샘플 추출을 단순화했습니다. (2차 패치에서 ‘유사 상태 매칭’으로 정교화 가능)")
    for r in view[:40]:
        row_card(r)

    if err:
        with st.expander("⚠️ 데이터/계산 실패 목록", expanded=False):
            for r in err:
                st.write(f"- {r.get('symbol')}: {r.get('error')}")

if __name__ == "__main__":
    main()
