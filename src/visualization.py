# visualization.py
# ============================================================
# 분석/임상/모델 비교까지 포함한 "완성형 시각화 모듈"
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 서버 환경용 (GUI 없이 저장)
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 공통 저장 유틸
# ------------------------------------------------------------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 00. SG vs BG Scatter
# ------------------------------------------------------------
def plot_scatter(df, results_dir):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(df["SG"], df["BG"], alpha=0.4)
    plt.xlabel("SG")
    plt.ylabel("BG")
    plt.title("SG vs BG Scatter")
    _save(fig, f"{results_dir}/00.scatter_sg_bg.png")


# ------------------------------------------------------------
# 01. Actual vs Predicted
# ------------------------------------------------------------
def plot_actual_vs_pred(y_true, y_pred, model_name, results_dir):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot(
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
def plot_residual(y_true, y_pred, model_name, results_dir):
    residual = y_pred - y_true
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residual, alpha=0.4)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Actual BG")
    plt.ylabel("Residual (Pred - True)")
    plt.title(f"Residual Plot ({model_name})")
    _save(fig, f"{results_dir}/02.residual_{model_name}.png")


# ------------------------------------------------------------
# 03. Bland–Altman Plot
# ------------------------------------------------------------
def plot_bland_altman(y_true, y_pred, model_name, results_dir):
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true

    m = np.mean(diff)
    sd = np.std(diff)

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(mean, diff, alpha=0.4)
    plt.axhline(m, linestyle="--", color="black", linewidth=2)
    plt.axhline(m + 1.96 * sd, linestyle=":", color="red", linewidth=1)
    plt.axhline(m - 1.96 * sd, linestyle=":", color="red", linewidth=1)
    plt.xlabel("Mean BG")
    plt.ylabel("Difference (Pred - True)")
    plt.title(f"Bland–Altman ({model_name})")
    _save(fig, f"{results_dir}/03.bland_altman_{model_name}.png")


# ------------------------------------------------------------
# 04. CEGA Zone Plot
# ------------------------------------------------------------
def plot_cega(y_true, y_pred, model_name, results_dir):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)

    # 기준선
    plt.plot([0, 400], [0, 400], "k--")

    # A-zone (±20%)
    plt.plot([0, 400], [0, 1.2 * 400], "k:")
    plt.plot([0, 400], [0, 0.8 * 400], "k:")

    # Zone 비율 계산
    diff_ratio = np.abs(y_pred - y_true) / y_true
    A = np.mean(diff_ratio <= 0.2) * 100
    B = np.mean(diff_ratio <= 0.3) * 100
    AB = B

    plt.text(
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
def plot_model_metrics(df_metrics, results_dir):
    metrics = ["r2", "rmse", "mae", "mard"]
    models = df_metrics["model"].tolist()
    colors = plt.cm.tab10.colors

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
# 06. K-Fold Cross-Validation Metrics Bar Plot
# ------------------------------------------------------------
def plot_kfold_metrics(kfold_df, results_dir):
    """
    kfold_df:
      columns = [model, fold, r2, rmse, mae, mard, ...]
    """

    metrics = ["r2", "rmse", "mae", "mard"]
    models = kfold_df["model"].unique()
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, metric in zip(axes, metrics):
        mean = kfold_df.groupby("model")[metric].mean()
        std = kfold_df.groupby("model")[metric].std()

        bars = ax.bar(
            mean.index,
            mean.values,
            yerr=std.values,
            capsize=4,
            color=colors[:len(mean)]
        )

        ax.set_title(f"{metric.upper()} (K-Fold)")
        ax.tick_params(axis="x", rotation=45)

        for bar, val in zip(bars, mean.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    _save(fig, f"{results_dir}/06.kfold_metrics_bar.png")


# ------------------------------------------------------------
# 07. SHAP Summary Plot (이미 계산된 경우)
# ------------------------------------------------------------
def save_shap_summary(shap_values, X, feature_names, results_dir):
    import shap

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(f"{results_dir}/07.shap_summary.png", dpi=200)
    plt.close()
