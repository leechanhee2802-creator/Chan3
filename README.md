# Angle Lab (회귀채널 + TP/SL 확률 스캐너)

## 목표
- 회귀(Regression) 채널로 "빗각 레일" 생성
- 신호 발생 시 과거 데이터로 TP/SL 먼저 도달 확률 계산
- LONG/SHORT/HOLD + STRONG/WEAK 분류
- 진입/손절/목표가(가격) + 근거(확률/샘플수/RR) 표시

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
