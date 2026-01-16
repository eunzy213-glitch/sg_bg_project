# tools/merge_visualizations.py
# ============================================================
# 목적:
# - 기존 파이프라인 결과(results/{experiment}/predictions.csv)를 기반으로
# - 모델별로 따로 저장된 시각화 "외에"
# - "모델 5개를 한 화면"에 모아 MERGE 폴더에 저장하는 후처리 스크립트
#
# 결과 저장 위치:
# - results/{experiment}/merge/
#
# 저장 파일명 규칙:
# - {experiment_lower}_merge_actual_vs_pred.png
# - {experiment_lower}_merge_residual.png
# - {experiment_lower}_merge_bland_altman.png
# - {experiment_lower}_merge_cega.png
# ============================================================

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 공통 유틸: predictions.csv 로드
# ------------------------------------------------------------
def load_predictions(results_root: str, experiment: str) -> pd.DataFrame:
    pred_path = os.path.join(results_root, experiment, "predictions.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ predictions.csv 파일이 없습니다: {pred_path}")

    df = pd.read_csv(pred_path)

    required_cols = ["model", "y_true", "y_pred"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"❌ predictions.csv에 '{c}' 컬럼이 없습니다.")

    if "residual" not in df.columns:
        df["residual"] = df["y_pred"] - df["y_true"]

    return df


def ensure_merge_dir(results_root: str, experiment: str) -> str:
    merge_dir = os.path.join(results_root, experiment, "merge")
    os.makedirs(merge_dir, exist_ok=True)
    return merge_dir


# ------------------------------------------------------------
# subplot grid 계산
# ------------------------------------------------------------
def compute_grid(n_items: int, ncols: int = 3):
    nrows = math.ceil(n_items / ncols)
    return nrows, ncols


# ------------------------------------------------------------
# 1) Actual vs Pred
# ------------------------------------------------------------
def plot_merge_actual_vs_pred(df, models, save_path):
    n = len(models)
    nrows, ncols = compute_grid(n, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    global_min = min(df["y_true"].min(), df["y_pred"].min())
    global_max = max(df["y_true"].max(), df["y_pred"].max())

    for i, model in enumerate(models):
        d = df[df["model"] == model]
        ax = axes[i]

        ax.scatter(d["y_true"], d["y_pred"], alpha=0.6)
        ax.plot([global_min, global_max], [global_min, global_max], "--", color="black")

        ax.set_title(f"Actual vs Pred ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Predicted BG")
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 2) Residual
# ------------------------------------------------------------
def plot_merge_residual(df, models, save_path):
    n = len(models)
    nrows, ncols = compute_grid(n, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, model in enumerate(models):
        d = df[df["model"] == model]
        ax = axes[i]

        ax.scatter(d["y_true"], d["residual"], alpha=0.6)
        ax.axhline(0, linestyle="--", color="black")

        ax.set_title(f"Residual ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Residual (Pred - True)")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 3) Bland–Altman (⭐ 여기 수정됨)
# ------------------------------------------------------------
def plot_merge_bland_altman(df, models, save_path):
    n = len(models)
    nrows, ncols = compute_grid(n, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, model in enumerate(models):
        d = df[df["model"] == model]
        ax = axes[i]

        mean_bg = (d["y_true"] + d["y_pred"]) / 2
        diff = d["y_pred"] - d["y_true"]

        mean_diff = diff.mean()
        sd_diff = diff.std()

        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff

        ax.scatter(mean_bg, diff, alpha=0.6)

        ax.axhline(mean_diff, linestyle="--", color="black", linewidth=2)
        ax.axhline(loa_upper, linestyle=":", color="red", linewidth=1)
        ax.axhline(loa_lower, linestyle=":", color="red", linewidth=1)

        # ⭐ CEGA 스타일 요약 텍스트 추가
        ax.text(
            0.98, 0.98,
            f"Mean: {mean_diff:.2f}\n"
            f"+1.96 SD: {loa_upper:.2f}\n"
            f"-1.96 SD: {loa_lower:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85)
        )

        ax.set_title(f"Bland–Altman ({model})")
        ax.set_xlabel("Mean of BG")
        ax.set_ylabel("Difference (Pred - True)")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 4) CEGA
# ------------------------------------------------------------
def plot_merge_cega(df, models, save_path):
    n = len(models)
    nrows, ncols = compute_grid(n, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    global_min = min(df["y_true"].min(), df["y_pred"].min())
    global_max = max(df["y_true"].max(), df["y_pred"].max())

    for i, model in enumerate(models):
        d = df[df["model"] == model]
        ax = axes[i]

        ax.scatter(d["y_true"], d["y_pred"], alpha=0.6)
        ax.plot([global_min, global_max], [global_min, global_max], "--", color="black")
        ax.plot([0, global_max], [0, global_max * 1.2], ":", color="gray")
        ax.plot([0, global_max], [0, global_max * 0.8], ":", color="gray")

        ratio = np.abs(d["y_pred"] - d["y_true"]) / d["y_true"].replace(0, np.nan)
        A = np.mean(ratio <= 0.2) * 100
        B_only = np.mean((ratio > 0.2) & (ratio <= 0.3)) * 100
        AB = np.mean(ratio <= 0.3) * 100

        ax.text(
            0.98, 0.98,
            f"A: {A:.1f}%\nB: {B_only:.1f}%\nA+B: {AB:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.2)
        )

        ax.set_title(f"CEGA ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Predicted BG")
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 실행 엔트리
# ------------------------------------------------------------
def main():
    results_root = "results"
    experiments = ["SG_ONLY", "SG_PLUS_META"]
    models = ["Linear", "Polynomial", "Huber", "RandomForest", "LightGBM"]

    for exp in experiments:
        df = load_predictions(results_root, exp)
        merge_dir = ensure_merge_dir(results_root, exp)
        exp_lower = exp.lower()

        plot_merge_actual_vs_pred(
            df, models,
            os.path.join(merge_dir, f"{exp_lower}_merge_actual_vs_pred.png")
        )
        plot_merge_residual(
            df, models,
            os.path.join(merge_dir, f"{exp_lower}_merge_residual.png")
        )
        plot_merge_bland_altman(
            df, models,
            os.path.join(merge_dir, f"{exp_lower}_merge_bland_altman.png")
        )
        plot_merge_cega(
            df, models,
            os.path.join(merge_dir, f"{exp_lower}_merge_cega.png")
        )

        print(f"✅ MERGE 저장 완료: {exp} → {merge_dir}")


if __name__ == "__main__":
    main()
