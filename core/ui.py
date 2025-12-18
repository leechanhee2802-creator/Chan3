import streamlit as st

def inject_css():
    st.markdown(
        """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, [data-testid="stAppViewContainer"] {
  font-family: "Pretendard", system-ui, -apple-system, sans-serif;
  background: linear-gradient(135deg, #f4f7ff 0%, #eefdfd 50%, #fdfcfb 100%);
  color: #0f172a;
}
main.block-container { max-width: 1300px; padding-top: 1rem; padding-bottom: 2rem; }

.card {
  background: rgba(255,255,255,0.96);
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
  margin-bottom: 10px;
}
.card-sm {
  background: rgba(255,255,255,0.96);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 10px 12px;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
  margin-bottom: 8px;
}
.chip { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.84rem; }
.chip-blue{ background:#dbeafe; color:#1d4ed8; }
.chip-green{ background:#bbf7d0; color:#166534; }
.chip-red{ background:#fee2e2; color:#b91c1c; }
.chip-gray{ background:#e5e7eb; color:#111827; }

.badge { font-weight:800; padding:6px 10px; border-radius:12px; display:inline-block; }
.badge-long{ background:#bbf7d0; color:#166534; }
.badge-short{ background:#fee2e2; color:#b91c1c; }
.badge-hold{ background:#e5e7eb; color:#111827; }

.small { color:#64748b; font-size:0.86rem; }
.big { font-size:1.25rem; font-weight:900; }

table { font-size:0.92rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

def badge(side: str, strength: str):
    side = side or "HOLD"
    strength = strength or "WEAK"
    if side == "LONG":
        cls = "badge badge-long"
    elif side == "SHORT":
        cls = "badge badge-short"
    else:
        cls = "badge badge-hold"
    return f'<span class="{cls}">{side} ({strength})</span>'

def fmt(x, digits=2):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"

def summary_cards(stats: dict):
    """
    상단 요약: 전체/롱/숏/홀드, STRONG 비율, 평균 확률 등
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="big">📌 스캐너 요약</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card-sm"><div class="small">총 종목</div><div class="big">{stats.get("total",0)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card-sm"><div class="small">LONG</div><div class="big">{stats.get("long",0)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card-sm"><div class="small">SHORT</div><div class="big">{stats.get("short",0)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card-sm"><div class="small">HOLD</div><div class="big">{stats.get("hold",0)}</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def row_card(item: dict):
    """
    각 종목 카드(캡처 느낌)
    """
    side = item.get("side", "HOLD")
    strength = item.get("strength", "WEAK")
    sym = item.get("symbol", "")
    price = item.get("price")
    score = item.get("score")
    wr = item.get("tp_sl_winrate")
    n = item.get("tp_sl_n")
    rr = item.get("rr")
    entry = item.get("entry")
    tp = item.get("tp")
    sl = item.get("sl")
    zlo = item.get("entry_zone_low")
    zhi = item.get("entry_zone_high")
    reason = item.get("reason", "")

    chip_cls = "chip-blue"
    if side == "LONG":
        chip_cls = "chip-green"
    elif side == "SHORT":
        chip_cls = "chip-red"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="chip {chip_cls}"><b>{sym}</b></span>
            {badge(side, strength)}
            <span class="chip chip-gray">Score {fmt(score,0)}</span>
          </div>
          <div class="small">현재가 {fmt(price)} USD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"TP/SL 성공확률: **{fmt(wr,1)}%**")
        st.caption(f"샘플수: {n if n is not None else '-'}")
    with c2:
        st.write(f"손익비(RR): **{fmt(rr,2)}**")
        st.caption("TP/SL % 기준")
    with c3:
        st.write(f"진입가: **{fmt(entry)}**")
        st.caption(f"권장구간: {fmt(zlo)} ~ {fmt(zhi)}")
    with c4:
        st.write(f"TP/SL: **{fmt(tp)} / {fmt(sl)}**")
        st.caption("TP=목표, SL=손절")

    st.caption(f"근거: {reason}")
    st.markdown('</div>', unsafe_allow_html=True)
