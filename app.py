import io
from typing import Dict, Tuple, List

import streamlit as st
import pandas as pd
from PIL import Image
import requests

# -----------------------------------------------------------
# 기본 페이지 설정
# -----------------------------------------------------------
st.set_page_config(
    page_title="Food & Macro Analyzer",
    page_icon="🥗",
    layout="wide",
)

# -----------------------------------------------------------
# 커스텀 CSS (깔끔 & 다크톤 대시보드 느낌)
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #f9fafb !important;
    }
    .stSidebar {
        background-color: #020617 !important;
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 1rem;
        background: #020617;
        border: 1px solid #1e293b;
    }
    .divider {
        border-bottom: 1px solid #1f2937;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# 음식 영양 DB (100g 기준 예시)
# kcal, carb, protein, fat (per 100g)
# -----------------------------------------------------------
FOOD_DB: Dict[str, Dict[str, float]] = {
    "백미밥":     {"kcal": 150, "carb": 34, "protein": 3,  "fat": 0.3},
    "현미밥":     {"kcal": 145, "carb": 31, "protein": 3.2,"fat": 1.0},
    "닭가슴살":   {"kcal": 165, "carb": 0,  "protein": 31, "fat": 3.6},
    "삶은 계란":  {"kcal": 155, "carb": 1.1,"protein": 13,"fat": 11},
    "고구마":     {"kcal": 86,  "carb": 20,"protein": 1.6,"fat": 0.1},
    "삼겹살":     {"kcal": 330, "carb": 0, "protein": 16,"fat": 30},
    "샐러드(드레싱 없음)": {"kcal": 25, "carb": 5, "protein": 1.5,"fat": 0.2},
    "식빵":       {"kcal": 250,"carb": 45,"protein": 8, "fat": 3},
    "떡볶이":     {"kcal": 180,"carb": 35,"protein": 4, "fat": 2},
    "김치찌개":   {"kcal": 80, "carb": 6, "protein": 5, "fat": 4},
}

HF_MODEL_ID = "nateraw/food101"  # 음식 특화 모델


# -----------------------------------------------------------
# HuggingFace Inference API 호출
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def call_hf_api(image_bytes: bytes, top_k: int = 5) -> List[Dict]:
    """
    HuggingFace Inference API로 Food-101 모델을 호출.
    st.secrets["HF_TOKEN"]이 있으면 Authorization 헤더에 사용.
    """
    token = st.secrets.get("HF_TOKEN", None)

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {"top_k": top_k}

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
        headers=headers,
        params=params,
        data=image_bytes,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    # 일부 모델은 {"error": "..."} 형식으로 줄 수도 있어서 처리
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    return data


def analyze_food_image(image: Image.Image, top_k: int = 5) -> List[Dict]:
    """
    업로드된 이미지를 Food-101 분류기로 분석하고
    상위 top_k개의 예측 결과를 반환.
    각 결과는 {"label": str, "score": float} 형태.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    try:
        preds = call_hf_api(image_bytes, top_k=top_k)
        # 예상 형식: [{"label": "...", "score": 0.98}, ...]
        if not isinstance(preds, list):
            return []
        return preds
    except Exception as e:
        st.error("이미지 인식 API 호출 중 오류가 발생했습니다. 나중에 다시 시도해 주세요.")
        st.write("디버그용 메시지:", str(e))
        return []


# -----------------------------------------------------------
# 음식 이름 + 그램 수로부터 영양 계산
# -----------------------------------------------------------
def calc_macros(food_name: str, grams: float) -> Dict[str, float]:
    base = FOOD_DB.get(food_name)
    if base is None or grams <= 0:
        return {"kcal": 0.0, "carb": 0.0, "protein": 0.0, "fat": 0.0}

    ratio = grams / 100.0
    return {
        "kcal": round(base["kcal"] * ratio, 1),
        "carb": round(base["carb"] * ratio, 1),
        "protein": round(base["protein"] * ratio, 1),
        "fat": round(base["fat"] * ratio, 1),
    }


# -----------------------------------------------------------
# 단백질 권장량 계산
# -----------------------------------------------------------
def calc_protein_recommendation(
    weight: float,
    goal: str,
) -> Tuple[float, float]:
    """
    goal:
        - "마른 체형 유지"
        - "보통 / 체중 유지"
        - "근육 증가"
    반환: (g/kg, total_g)
    """
    if goal == "마른 체형 유지":
        factor = 1.4
    elif goal == "근육 증가":
        factor = 2.0
    else:  # "보통 / 체중 유지"
        factor = 1.6

    total_g = round(weight * factor, 1)
    return factor, total_g


# -----------------------------------------------------------
# 사이드바: 고객 정보 & 단백질 권장량
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 고객 프로필")
    name = st.text_input("이름(선택)", value="")
    age = st.number_input("나이", min_value=10, max_value=99, value=30)
    sex = st.selectbox("성별", ["남성", "여성"])
    height = st.number_input("키 (cm)", min_value=120, max_value=220, value=170)
    weight = st.number_input("몸무게 (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    goal = st.selectbox(
        "목표",
        ["마른 체형 유지", "보통 / 체중 유지", "근육 증가"],
    )

    factor, protein_total = calc_protein_recommendation(weight, goal)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🧬 단백질 권장량")

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.9rem;color:#9ca3af;">단백질 권장 기준</div>
            <div style="font-size:1.2rem;font-weight:600;margin-top:0.2rem;">
                {factor} g / kg
            </div>
            <div style="margin-top:0.6rem;font-size:0.9rem;color:#9ca3af;">
                목표 👉 <b>{goal}</b>
            </div>
            <div style="margin-top:0.4rem;font-size:1.0rem;">
                하루 권장 단백질: <b>{protein_total} g</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
# 메인 영역
# -----------------------------------------------------------
st.markdown("## 🥗 Food & Macro Analyzer")
st.markdown(
    "업로드한 **식사 사진**을 기반으로 AI가 음식 종류를 추정하고, "
    "음식 정보를 입력하면 대략적인 칼로리와 탄수화물·단백질·지방을 계산합니다."
)

col_img, col_form = st.columns([1.1, 1.4])

# ------------------ 사진 업로드 & AI 예측 ------------------
preds_text = ""

with col_img:
    st.markdown("### 📸 식사 사진 업로드")
    uploaded_file = st.file_uploader(
        "사진을 업로드하세요 (jpg/png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="업로드된 사진", use_column_width=True)

        with st.spinner("🍽️ 음식 인식 중... (Food-101 모델)"):
            preds = analyze_food_image(image, top_k=5)

        if preds:
            lines = []
            for p in preds:
                label = str(p.get("label", "")).replace("_", " ")
                score = p.get("score", 0.0)
                score_pct = round(float(score) * 100, 1)
                lines.append(f"- {label} ({score_pct}%)")
            preds_text = "\n".join(lines)

            st.markdown("#### 🔍 AI가 추측한 음식 (Top-5)")
            st.markdown(
                f"<pre style='background:#020617;padding:0.75rem;border-radius:0.5rem;border:1px solid #1f2937;font-size:0.85rem;'>{preds_text}</pre>",
                unsafe_allow_html=True,
            )
            st.caption("※ 영어로 나온 음식 이름은 참고용입니다. 실제 선택은 아래 폼에서 직접 입력/선택해 주세요.")
        else:
            st.warning("AI 예측 결과를 가져오지 못했습니다. 나중에 다시 시도해 주세요.")


# ------------------ 음식 입력 & 영양 계산 ------------------
with col_form:
    st.markdown("### 🍽️ 음식 구성 입력")

    st.markdown(
        "<small style='color:#9ca3af;'>"
        "1️⃣ 왼쪽에서 사진을 업로드하면 AI가 음식 후보를 보여줍니다.<br>"
        "2️⃣ 아래에서 실제 먹은 음식 이름과 양(g)을 입력하면 칼로리와 3대 영양소를 계산합니다.<br>"
        "※ 현재 버전은 g(그램) 추정은 사용자가 직접 입력해야 합니다."
        "</small>",
        unsafe_allow_html=True,
    )

    food_rows = []

    if "num_rows" not in st.session_state:
        st.session_state["num_rows"] = 3

    num_rows = st.session_state["num_rows"]

    with st.form("food_form"):
        for i in range(num_rows):
            st.markdown(f"##### 음식 #{i+1}")
            col1, col2, col3 = st.columns([1.3, 1.0, 1.2])
            with col1:
                food_name = st.selectbox(
                    "음식 이름 (DB 선택)",
                    options=["(선택 안 함)"] + list(FOOD_DB.keys()),
                    key=f"food_name_{i}",
                )
            with col2:
                grams = st.number_input(
                    "양 (g)",
                    min_value=0.0,
                    max_value=2000.0,
                    value=0.0,
                    step=10.0,
                    key=f"grams_{i}",
                )
            with col3:
                custom_name = st.text_input(
                    "직접 이름 입력 (선택)",
                    value="",
                    key=f"custom_name_{i}",
                    help="DB에 없는 음식은 여기에 한글/영어로 적어두면 기록용으로 표시됩니다.",
                )

            final_name = custom_name.strip() if custom_name.strip() else food_name
            food_rows.append((final_name, grams))

            st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            add_row = st.form_submit_button("➕ 음식 줄 추가")
        with col_btn2:
            submit = st.form_submit_button("🔍 분석하기")

        if add_row:
            st.session_state["num_rows"] = num_rows + 1

    # ------------------ 분석 결과 ------------------
    total_kcal = 0.0
    total_carb = 0.0
    total_protein = 0.0
    total_fat = 0.0

    result_rows = []
    for (name, grams) in food_rows:
        if not name or name == "(선택 안 함)" or grams <= 0:
            continue

        macros = calc_macros(name if name in FOOD_DB else "", grams)
        total_kcal += macros["kcal"]
        total_carb += macros["carb"]
        total_protein += macros["protein"]
        total_fat += macros["fat"]

        result_rows.append(
            {
                "음식": name,
                "양(g)": grams,
                "칼로리(kcal)": macros["kcal"],
                "탄수화물(g)": macros["carb"],
                "단백질(g)": macros["protein"],
                "지방(g)": macros["fat"],
            }
        )

    if result_rows:
        st.markdown("### ✅ 식단 영양 분석 결과")
        df = pd.DataFrame(result_rows)
        st.dataframe(
            df.style.format(
                {
                    "양(g)": "{:.0f}",
                    "칼로리(kcal)": "{:.1f}",
                    "탄수화물(g)": "{:.1f}",
                    "단백질(g)": "{:.1f}",
                    "지방(g)": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("총 칼로리", f"{round(total_kcal, 1)} kcal")
        col_b.metric("총 탄수화물", f"{round(total_carb, 1)} g")
        col_c.metric("총 단백질", f"{round(total_protein, 1)} g")
        col_d.metric("총 지방", f"{round(total_fat, 1)} g")

        if protein_total > 0:
            ratio = round(total_protein / protein_total * 100, 1)
            st.markdown(
                f"💪 오늘 식사의 단백질 섭취량은 **권장량의 약 {ratio}%** 입니다."
            )
    else:
        st.markdown("아직 유효한 음식 입력이 없습니다. 음식 이름과 g(그램)을 입력해 보세요.")
