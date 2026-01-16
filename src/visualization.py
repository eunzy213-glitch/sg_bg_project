# visualization.py
# ============================================================
# 분석/임상/모델 비교까지 포함한 "완성형 시각화 모듈"
# ============================================================

import os # 파일/폴더 경로 라이브러리
import numpy as np # 수치계산 라이브러리
import pandas as pd # DataFrame 처리 라이브러리
import matplotlib # 시각화 라이브러리
matplotlib.use("Agg")   # GUI 없는 서버 환경에서도 그림 저장 가능하도록 설정
import matplotlib.pyplot as plt # 시각화 라이브러리

# ------------------------------------------------------------
# 공통 저장 유틸 함수
# ------------------------------------------------------------
def _save(fig, path): # 모든 시각화 함수에서 공통으로 사용하는 저장함수
    fig.tight_layout() # subplot 간격 자동 조정
    fig.savefig(path, dpi=200) # 해상도 200으로 저장
    plt.close(fig) # 메모리 절약을 위해 닫기

# ------------------------------------------------------------
# 00. SG vs BG Scatter
# ------------------------------------------------------------
def plot_scatter(df, results_dir): # 원본 데이터에서 SG와 BG의 산점도
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(df["SG"], df["BG"], alpha=0.4)
    plt.xlabel("SG")
    plt.ylabel("BG")
    plt.title("SG vs BG Scatter")
    _save(fig, f"{results_dir}/00.scatter_sg_bg.png")

# ------------------------------------------------------------
# 01. Actual vs Predicted
# ------------------------------------------------------------
def plot_actual_vs_pred(y_true, y_pred, model_name, results_dir): # 실제 BG vs 예측 BG 산점도
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot( # 이상적인 예측선
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--",
        color="red"
    )
    plt.xlabel("Actual BG")
    plt.ylabel("Predicted BG")
    plt.title(f"Actual vs Predicted ({model_name})")
    _save(fig, f"{results_dir}/01.actual_vs_pred_{model_name}.png")


# ------------------------------------------------------------
# 02. Residual Plot
# ------------------------------------------------------------
def plot_residual(y_true, y_pred, model_name, results_dir): # 잔차 분석 그래프
    residual = y_pred - y_true
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residual, alpha=0.4)
    plt.axhline(0, linestyle="--", color="black") # 잔차 0 기준선
    plt.xlabel("Actual BG")
    plt.ylabel("Residual (Pred - True)")
    plt.title(f"Residual Plot ({model_name})")
    _save(fig, f"{results_dir}/02.residual_{model_name}.png")

# ------------------------------------------------------------
# 03. Bland–Altman Plot
# ------------------------------------------------------------
def plot_bland_altman(y_true, y_pred, model_name, results_dir):
    # 평균과 차이 계산
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true

    # 통계값 계산
    mean_diff = np.mean(diff)                 # 평균 bias
    sd_diff = np.std(diff, ddof=1)            # 표준편차
    loa_upper = mean_diff + 1.96 * sd_diff    # 상한 LoA
    loa_lower = mean_diff - 1.96 * sd_diff    # 하한 LoA

    fig = plt.figure(figsize=(6, 4))

    # 산점도
    plt.scatter(mean, diff, alpha=0.4)

    # 기준선들
    plt.axhline(mean_diff, linestyle="--", color="black", linewidth=2)
    plt.axhline(loa_upper, linestyle=":", color="red", linewidth=1)
    plt.axhline(loa_lower, linestyle=":", color="red", linewidth=1)

    # 오른쪽 위 요약 텍스트 박스
    plt.text(
        0.95, 0.95,
        f"Mean bias: {mean_diff:.2f}\n"
        f"+1.96 SD: {loa_upper:.2f}\n"
        f"-1.96 SD: {loa_lower:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.xlabel("Mean BG")
    plt.ylabel("Difference (Pred - True)")
    plt.title(f"Bland–Altman ({model_name})")

    _save(fig, f"{results_dir}/03.bland_altman_{model_name}.png")

# ------------------------------------------------------------
# 04. CEGA Zone Plot
# ------------------------------------------------------------
def plot_cega(y_true, y_pred, model_name, results_dir): # CEGA 분석
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4) # 실제 vs 예측 산점도

    # 기준선
    plt.plot([0, 400], [0, 400], "k--")

    # A-zone (±20%)
    plt.plot([0, 400], [0, 1.2 * 400], "k:")
    plt.plot([0, 400], [0, 0.8 * 400], "k:")

    # A+B Zone 비율 계산
    diff_ratio = np.abs(y_pred - y_true) / y_true
    A = np.mean(diff_ratio <= 0.2) * 100
    B = np.mean(diff_ratio <= 0.3) * 100
    AB = B

    plt.text( # 그래프 내 텍스트 박스
        0.05, 0.95,
        f"A zone: {A:.1f}%\nB zone: {B:.1f}%\nA+B: {AB:.1f}%",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.xlabel("Actual BG")
    plt.ylabel("Predicted BG")
    plt.title(f"CEGA ({model_name})")

    _save(fig, f"{results_dir}/04.cega_{model_name}.png")

# ------------------------------------------------------------
# 05. Model Performance Bar Plot (R2 / RMSE / MAE / MARD)
# ------------------------------------------------------------
def plot_model_metrics(df_metrics, results_dir): # 모델별 성능지표를 한 화면에서 비교하는 bar plot
    metrics = ["r2", "rmse", "mae", "mard"] # 시각화할 평가지표 목록
    models = df_metrics["model"].tolist() # 모델이름
    colors = plt.cm.tab10.colors # 색상 팔레트

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, metric in zip(axes, metrics):
        values = df_metrics[metric].values

        bars = ax.bar(
            models,
            values,
            color=colors[:len(models)]
        )

        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)

        # 막대 위에 숫자 표시
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    _save(fig, f"{results_dir}/05.model_metrics_bar.png")

# ------------------------------------------------------------
# 06. SHAP Summary Plot
# ------------------------------------------------------------
def save_shap_summary(shap_values, X, feature_names, results_dir): # SHAP summary plot 저장
    import shap

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(f"{results_dir}/07.shap_summary.png", dpi=200)
    plt.close()
