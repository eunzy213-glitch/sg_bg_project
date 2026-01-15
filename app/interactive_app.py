# interactive_app.py
# ============================================================
# SG → BG Prediction Interactive Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------------------------------------
# Streamlit 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="SG → BG Prediction Dashboard",
    layout="wide"
)

st.title("🧪 SG → BG Prediction Analysis Dashboard")

# ------------------------------------------------------------
# 1️⃣ 실험 / 모델 선택
# ------------------------------------------------------------
experiment = st.sidebar.selectbox(
    "Experiment",
    ["SG_ONLY", "SG_PLUS_META"]
)

model = st.sidebar.selectbox(
    "Model",
    ["Linear", "Polynomial", "Huber", "RandomForest", "LightGBM"]
)

# ------------------------------------------------------------
# 2️⃣ 데이터 로드
# ------------------------------------------------------------
data_path = f"results/{experiment}/predictions.csv"

if not os.path.exists(data_path):
    st.error(f"❌ {data_path} 파일이 없습니다.")
    st.stop()

df = pd.read_csv(data_path)

# 모델 필터링
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
    "🧠 CEGA"
])

# ============================================================
# 📈 Actual vs Predicted
# ============================================================
with tabs[0]:
    st.subheader("Actual vs Predicted BG")

    fig = px.scatter(
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
with tabs[1]:
    st.subheader("Residual Plot")

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
with tabs[2]:
    st.subheader("Bland–Altman Plot")

    mean_bg = (y_true + y_pred) / 2
    diff = y_pred - y_true

    mean_diff = diff.mean()
    sd_diff = diff.std()

    fig = px.scatter(
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

    # 평균 차이선
    fig.add_hline(
        y=mean_diff,
        line_color="black",
        line_dash="dash"
    )

    # ±1.96 SD
    fig.add_hline(
        y=mean_diff + 1.96 * sd_diff,
        line_color="red",
        line_dash="dot"
    )
    fig.add_hline(
        y=mean_diff - 1.96 * sd_diff,
        line_color="red",
        line_dash="dot"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🧠 CEGA Plot
# ============================================================
with tabs[3]:
    st.subheader("Clarke Error Grid Analysis (CEGA)")

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

    # y = x 기준선
    fig.add_shape(
        type="line",
        x0=min_bg, y0=min_bg,
        x1=max_bg, y1=max_bg,
        line=dict(dash="dash", color="black")
    )

    # A-zone ±20%
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_bg, y1=max_bg * 1.2,
        line=dict(dash="dot", color="gray")
    )
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_bg, y1=max_bg * 0.8,
        line=dict(dash="dot", color="gray")
    )

    # Zone 비율 계산
    ratio = np.abs(y_pred - y_true) / y_true.replace(0, np.nan)

    A = np.mean(ratio <= 0.2) * 100
    B = np.mean((ratio > 0.2) & (ratio <= 0.3)) * 100
    AB = np.mean(ratio <= 0.3) * 100

    st.markdown(
        f"""
        **A zone:** {A:.1f}%  
        **B zone:** {B:.1f}%  
        **A + B zone:** {AB:.1f}%
        """
    )

    st.plotly_chart(fig, use_container_width=True)
