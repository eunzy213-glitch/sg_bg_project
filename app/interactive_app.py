# interactive_app.py
# ============================================================
# SG → BG Prediction Interactive Dashboard
# 모델별 시각화를 인터랙티브하게 확인하는 앱
# ============================================================

import streamlit as st # Streamlit 라이브러리
import pandas as pd # csv 파일 로드 및 DataFrame 처리
import numpy as np # 수치 계산
import plotly.express as px # 시각화 라이브러리
import plotly.graph_objects as go # 시각화 라이브러리
import os # 파일 존재 여부 확인/경로 처리

# ------------------------------------------------------------
# Streamlit 기본 설정
# ------------------------------------------------------------
st.set_page_config( # Streamlit 페이지 기본 설정
    page_title="SG → BG Prediction Dashboard",
    layout="wide"
)

st.title("🧪 SG → BG Prediction Analysis Dashboard")

# ------------------------------------------------------------
# 1️⃣ 실험 선택
# ------------------------------------------------------------
experiment = st.sidebar.selectbox( # 사이드바에 드롭다운 생성
    "Experiment", # 드롭다운 위에 표시될 라벨
    ["SG_ONLY", "SG_PLUS_META"] # 선택 가능한 실험 이름
)

# ------------------------------------------------------------
# 2️⃣ 데이터 로드
# ------------------------------------------------------------
data_path = f"results/{experiment}/predictions.csv" # 선택한 experiment에 해당하는 예측 결과 CSV 경로 생성

if not os.path.exists(data_path):
    st.error(f"❌ {data_path} 파일이 없습니다.")
    st.stop()

df = pd.read_csv(data_path)

# ------------------------------------------------------------
# 🆕 모델 목록 자동 추출 (추가)
# ------------------------------------------------------------
available_models = sorted(df["model"].unique().tolist())

model = st.sidebar.selectbox( # 사이드바 두번째 드롭다운
    "Model", # 라벨
    available_models
)

# ------------------------------------------------------------
# 모델 필터링
# ------------------------------------------------------------
df_model = df[df["model"] == model].copy()

if df_model.empty:
    st.warning("선택한 모델에 대한 데이터가 없습니다.")
    st.stop()

# ------------------------------------------------------------
# 공통 변수 (Series 형태 유지)
# ------------------------------------------------------------
y_true = df_model["y_true"]
y_pred = df_model["y_pred"]
residual = y_pred - y_true
sg = df_model["SG"]

# ------------------------------------------------------------
# 3️⃣ 탭 구성
# ------------------------------------------------------------
tabs = st.tabs([
    "📈 Actual vs Predicted",
    "📉 Residual",
    "📊 Bland–Altman",
    "🧠 CEGA",
    "🧩 SEG Analysis" # ✅ SEG 탭 추가
])

# ============================================================
# 📈 Actual vs Predicted
# ============================================================
with tabs[0]: # tabs[0] 영역에 그릴 UI 차트들
    st.subheader("Actual vs Predicted BG") # 탭 내부 소제목

    fig = px.scatter( # plotly express 산점도 생성
        df_model,
        x="y_true",                # ✅ 문자열 컬럼명
        y="y_pred",                # ✅ 문자열 컬럼명
        hover_data=["SG", "residual"],
        labels={
            "y_true": "Actual BG",
            "y_pred": "Predicted BG"
        },
        title=f"Actual vs Predicted BG ({model})"
    )

    # y = x 기준선
    min_bg = min(y_true.min(), y_pred.min())
    max_bg = max(y_true.max(), y_pred.max())

    fig.add_shape(
        type="line",
        x0=min_bg, y0=min_bg,
        x1=max_bg, y1=max_bg,
        line=dict(dash="dash", color="black")
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📉 Residual Plot
# ============================================================
with tabs[1]: # tabs[1] 영역
    st.subheader("Residual Plot") # 탭 내부 소제목

    fig = px.scatter(
        df_model,
        x="y_true",
        y="residual",
        hover_data=["SG"],
        labels={
            "y_true": "Actual BG",
            "residual": "Residual (Predicted - Actual)"
        },
        title=f"Residual Plot ({model})"
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📊 Bland–Altman Plot
# ============================================================
with tabs[2]: # tabs[2] 영역
    st.subheader("Bland–Altman Plot") # 탭 내부 소제목

    # --------------------------------------------------
    # 1️⃣ Bland–Altman 계산
    # --------------------------------------------------
    mean_bg = (y_true + y_pred) / 2          # (Actual + Predicted) / 2
    diff = y_pred - y_true                   # Difference = Predicted - Actual

    mean_diff = diff.mean()                  # 평균 편향 (bias)
    sd_diff = diff.std()                     # 차이의 표준편차

    loa_upper = mean_diff + 1.96 * sd_diff   # 상한 (Upper LoA)
    loa_lower = mean_diff - 1.96 * sd_diff   # 하한 (Lower LoA)

    # --------------------------------------------------
    # 2️⃣ Scatter Plot
    # --------------------------------------------------
    fig = px.scatter( # Bland-Altman 산점도 생성
        x=mean_bg,
        y=diff,
        hover_data={
            "Actual BG": y_true,
            "Predicted BG": y_pred
        },
        labels={
            "x": "Mean of BG",
            "y": "Difference (Predicted - Actual)"
        },
        title=f"Bland–Altman Plot ({model})"
    )

    # --------------------------------------------------
    # 3️⃣ 기준선 추가
    # --------------------------------------------------
    fig.add_hline(y=mean_diff, line_color="black", line_dash="dash")
    fig.add_hline(y=loa_upper, line_color="red", line_dash="dot")
    fig.add_hline(y=loa_lower, line_color="red", line_dash="dot")

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧠 CEGA Plot
# ============================================================
with tabs[3]: # tabs[3] 영역
    st.subheader("Clarke Error Grid Analysis (CEGA)") # 탭 내부 소제목

    fig = px.scatter(
        df_model,
        x="y_true",
        y="y_pred",
        hover_data=["SG"],
        labels={
            "y_true": "Actual BG",
            "y_pred": "Predicted BG"
        },
        title=f"CEGA Plot ({model})"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧩 SEG Analysis Tab (정리 완료)
# ============================================================
with tabs[4]: # tabs[4] 영역
    st.subheader("Surveillance Error Grid (SEG) Analysis") # 탭 내부 소제목

    # ------------------------------------------------------------
    # 1️⃣ SEG Detailed 결과 CSV 기반 Interactive Scatter
    # ------------------------------------------------------------
    st.markdown("### ✅ Interactive SEG Scatter (Hover Supported)")

    # ✅ 모델별 detailed SEG 결과 CSV 경로 탐색
    model_dir = f"results/{experiment}/{model}"
    detailed_csv_path = None

    if os.path.exists(model_dir):
        detailed_candidates = [
            f for f in os.listdir(model_dir)
            if f.lower().startswith("detailed_results") and f.lower().endswith(".csv")
        ]
        if len(detailed_candidates) > 0:
            detailed_csv_path = os.path.join(model_dir, detailed_candidates[0])

    if detailed_csv_path is None or not os.path.exists(detailed_csv_path):
        st.warning("❌ SEG detailed 결과 CSV 파일이 존재하지 않습니다.")
        st.stop()

    # ------------------------------------------------------------
    # 2️⃣ Detailed CSV 로드
    # ------------------------------------------------------------
    seg_detail = pd.read_csv(detailed_csv_path)

    # ------------------------------------------------------------
    # 3️⃣ Interactive Scatter Plot 생성 (Plotly)
    # ------------------------------------------------------------
    fig = px.scatter(
        seg_detail,
        x="Reference_BG",
        y="Predicted_BG",
        color="SEG_Zone",
        hover_data=[
            "Reference_BG",
            "Predicted_BG",
            "Absolute_Error",
            "Relative_Error_%",
            "SEG_Zone"
        ],
        labels={
            "Reference_BG": "Actual BG",
            "Predicted_BG": "Predicted BG",
            "SEG_Zone": "SEG Zone"
        },
        title=f"Interactive SEG Scatter ({model})"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # 2️⃣ 전체 모델 SEG Summary CSV 출력
    # ------------------------------------------------------------
    st.markdown("### ✅ Combined SEG Summary Table")

    summary_path = f"results/{experiment}/combined_summary_with_seg.csv"

    if os.path.exists(summary_path):
        seg_df = pd.read_csv(summary_path)
        st.dataframe(seg_df)

    else:
        st.warning("❌ combined_summary_with_seg.csv 파일이 존재하지 않습니다.")
