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
# 저장 파일명 규칙(요청 반영):
# - {experiment_lower}_merge_actual_vs_pred.png
# - {experiment_lower}_merge_residual.png
# - {experiment_lower}_merge_bland_altman.png
# - {experiment_lower}_merge_cega.png
# ============================================================

import os                           # 경로 처리 및 폴더 생성에 사용
import math                         # 그리드 행/열 계산(ceil 등)에 사용
import pandas as pd                 # predictions.csv 로드에 사용
import numpy as np                  # 수치 계산(평균, 표준편차 등)에 사용
import matplotlib.pyplot as plt     # 그림 생성/저장에 사용


# ------------------------------------------------------------
# 공통 유틸: MERGE 폴더 생성 + 실험 결과 데이터 로드
# ------------------------------------------------------------
def load_predictions(results_root: str, experiment: str) -> pd.DataFrame:
    """
    results/{experiment}/predictions.csv 파일을 읽어 DataFrame으로 반환합니다.

    Parameters
    ----------
    results_root : str
        results 폴더의 루트 경로 (기본: "results")
    experiment : str
        "SG_ONLY" 또는 "SG_PLUS_META"

    Returns
    -------
    pd.DataFrame
        predictions.csv를 읽은 데이터프레임
    """
    # predictions.csv 경로 생성
    pred_path = os.path.join(results_root, experiment, "predictions.csv")

    # 파일이 없으면 에러를 명확히 보여주기 위해 예외 발생
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ predictions.csv 파일이 없습니다: {pred_path}")

    # CSV 로드
    df = pd.read_csv(pred_path)

    # 필수 컬럼 확인(없으면 이후 플롯이 깨지므로 미리 체크)
    required_cols = ["model", "y_true", "y_pred"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"❌ predictions.csv에 '{c}' 컬럼이 없습니다.")

    # residual 컬럼이 없으면 여기서 만들어 둠(후처리에서 안전)
    if "residual" not in df.columns:
        df["residual"] = df["y_pred"] - df["y_true"]

    return df


def ensure_merge_dir(results_root: str, experiment: str) -> str:
    """
    results/{experiment}/merge 폴더를 생성하고 경로를 반환합니다.
    """
    # merge 폴더 경로 생성
    merge_dir = os.path.join(results_root, experiment, "merge")

    # 폴더가 없으면 생성
    os.makedirs(merge_dir, exist_ok=True)

    return merge_dir


# ------------------------------------------------------------
# 공통 유틸: "모델 N개를 한 화면" 레이아웃 계산
# ------------------------------------------------------------
def compute_grid(n_items: int, ncols: int = 3):
    """
    n_items(모델 개수)에 맞춰 subplot 그리드(행, 열)를 계산합니다.
    - 기본은 열 3개(보기 좋게)로 고정
    """
    # 행 개수 = 올림(n_items / ncols)
    nrows = math.ceil(n_items / ncols)

    return nrows, ncols


# ------------------------------------------------------------
# 1) Actual vs Pred: 모델 5개를 한 화면에
# ------------------------------------------------------------
def plot_merge_actual_vs_pred(
    df: pd.DataFrame,
    models: list,
    save_path: str
):
    """
    모델별 Actual vs Predicted 산점도를 한 화면으로 저장합니다.
    """
    # 모델 개수
    n = len(models)

    # 그리드 계산(기본 3열)
    nrows, ncols = compute_grid(n, ncols=3)

    # figure 크기(행/열에 비례해 적당히 키움)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # axes가 2차원 배열이 아닐 수 있어 flatten으로 1차원으로 통일
    axes = np.array(axes).reshape(-1)

    # 전체 범위(기준선 y=x를 모델별로 동일 범위로 보이게 하기 위함)
    global_min = min(df["y_true"].min(), df["y_pred"].min())
    global_max = max(df["y_true"].max(), df["y_pred"].max())

    # 모델별로 반복
    for i, model in enumerate(models):
        # 해당 모델 데이터 필터링
        d = df[df["model"] == model]

        # 현재 subplot 축 선택
        ax = axes[i]

        # 산점도: (Actual, Pred)
        ax.scatter(d["y_true"], d["y_pred"], alpha=0.6)

        # 기준선 y=x
        ax.plot([global_min, global_max], [global_min, global_max], linestyle="--", color="black")

        # 제목/축 라벨
        ax.set_title(f"Actual vs Pred ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Predicted BG")

        # 동일 스케일로 비교하기 위해 축 범위 고정
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

    # 남는 subplot이 있으면 빈 칸 숨김(예: 5개를 3x2로 만들면 1칸 남음)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 레이아웃 자동 정리
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 2) Residual plot: 모델 5개를 한 화면에
# ------------------------------------------------------------
def plot_merge_residual(
    df: pd.DataFrame,
    models: list,
    save_path: str
):
    """
    모델별 Residual plot(y_true vs residual)을 한 화면으로 저장합니다.
    """
    # 모델 개수
    n = len(models)

    # 그리드 계산
    nrows, ncols = compute_grid(n, ncols=3)

    # figure 생성
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # axes를 1차원으로 통일
    axes = np.array(axes).reshape(-1)

    # 모델별 반복
    for i, model in enumerate(models):
        # 모델 데이터 필터링
        d = df[df["model"] == model]

        # subplot 축 선택
        ax = axes[i]

        # 산점도: (Actual, Residual)
        ax.scatter(d["y_true"], d["residual"], alpha=0.6)

        # 잔차 0 기준선(검정 점선 요청 반영)
        ax.axhline(0, linestyle="--", color="black")

        # 제목/축 라벨
        ax.set_title(f"Residual ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Residual (Pred - True)")

    # 남는 subplot 숨김
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 레이아웃 정리
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 3) Bland-Altman: 모델 5개를 한 화면에
# ------------------------------------------------------------
def plot_merge_bland_altman(
    df: pd.DataFrame,
    models: list,
    save_path: str
):
    """
    모델별 Bland–Altman plot(Mean vs Diff)을 한 화면으로 저장합니다.
    """
    # 모델 개수
    n = len(models)

    # 그리드 계산
    nrows, ncols = compute_grid(n, ncols=3)

    # figure 생성
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # axes 1차원 통일
    axes = np.array(axes).reshape(-1)

    # 모델별 반복
    for i, model in enumerate(models):
        # 모델 데이터
        d = df[df["model"] == model]

        # subplot 축
        ax = axes[i]

        # mean, diff 계산
        mean_bg = (d["y_true"] + d["y_pred"]) / 2
        diff = d["y_pred"] - d["y_true"]

        # 평균/표준편차
        m = diff.mean()
        sd = diff.std()

        # 산점도
        ax.scatter(mean_bg, diff, alpha=0.6)

        # 가운데 굵은 점선: 검정(요청 반영)
        ax.axhline(m, linestyle="--", color="black", linewidth=2)

        # 위/아래 가는 점선: 빨강(요청 반영)
        ax.axhline(m + 1.96 * sd, linestyle=":", color="red", linewidth=1)
        ax.axhline(m - 1.96 * sd, linestyle=":", color="red", linewidth=1)

        # 제목/축 라벨
        ax.set_title(f"Bland–Altman ({model})")
        ax.set_xlabel("Mean of BG")
        ax.set_ylabel("Difference (Pred - True)")

    # 남는 subplot 숨김
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 레이아웃 정리
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 4) CEGA: 모델 5개를 한 화면에
# ------------------------------------------------------------
def plot_merge_cega(
    df: pd.DataFrame,
    models: list,
    save_path: str
):
    """
    모델별 CEGA(Clarke Error Grid Analysis) 형태의 산점도를 한 화면으로 저장합니다.
    - 여기서는 간단 CEGA 형태(기준선 + A-zone ±20%)를 동일하게 그립니다.
    """
    # 모델 개수
    n = len(models)

    # 그리드 계산
    nrows, ncols = compute_grid(n, ncols=3)

    # figure 생성
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    # axes 1차원 통일
    axes = np.array(axes).reshape(-1)

    # 전체 축 범위(모델 비교를 위해 통일)
    global_min = min(df["y_true"].min(), df["y_pred"].min())
    global_max = max(df["y_true"].max(), df["y_pred"].max())

    # 모델별 반복
    for i, model in enumerate(models):
        # 모델 데이터
        d = df[df["model"] == model]

        # subplot 축
        ax = axes[i]

        # 산점도
        ax.scatter(d["y_true"], d["y_pred"], alpha=0.6)

        # 기준선 y=x
        ax.plot([global_min, global_max], [global_min, global_max], linestyle="--", color="black")

        # A-zone ±20% (점선)
        ax.plot([0, global_max], [0, global_max * 1.2], linestyle=":", color="gray")
        ax.plot([0, global_max], [0, global_max * 0.8], linestyle=":", color="gray")

        # zone 비율(단순 비율 기반 표시: A(<=20%), B(<=30%), A+B(<=30%))
        ratio = np.abs(d["y_pred"] - d["y_true"]) / d["y_true"].replace(0, np.nan)
        A = np.mean(ratio <= 0.2) * 100
        B_only = np.mean((ratio > 0.2) & (ratio <= 0.3)) * 100
        AB = np.mean(ratio <= 0.3) * 100

        # 오른쪽 위에 텍스트로 표시
        ax.text(
            0.98, 0.98,
            f"A: {A:.1f}%\nB: {B_only:.1f}%\nA+B: {AB:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.2)
        )

        # 제목/축 라벨
        ax.set_title(f"CEGA ({model})")
        ax.set_xlabel("Actual BG")
        ax.set_ylabel("Predicted BG")

        # 축 범위 통일
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

    # 남는 subplot 숨김
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 레이아웃 정리
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 실행 엔트리: 두 실험 모두 MERGE 이미지 생성
# ------------------------------------------------------------
def main():
    """
    - SG_ONLY / SG_PLUS_META 두 실험에 대해
    - 모델 5개를 한 화면에 모은 MERGE 이미지를 생성합니다.
    """
    # results 폴더 루트(프로젝트 루트 기준)
    results_root = "results"

    # 실험 목록(필요하면 여기만 수정하면 됨)
    experiments = ["SG_ONLY", "SG_PLUS_META"]

    # 모델 목록(현재 프로젝트 모델 이름과 predictions.csv의 model 값이 동일해야 함)
    models = ["Linear", "Polynomial", "Huber", "RandomForest", "LightGBM"]

    # 실험별로 반복
    for exp in experiments:
        # predictions.csv 로드
        df = load_predictions(results_root, exp)

        # merge 폴더 생성
        merge_dir = ensure_merge_dir(results_root, exp)

        # 실험명 소문자(파일명 규칙 요청 반영)
        exp_lower = exp.lower()

        # 1) actual vs pred 저장 경로
        save_actual = os.path.join(merge_dir, f"{exp_lower}_merge_actual_vs_pred.png")

        # 2) residual 저장 경로
        save_resid = os.path.join(merge_dir, f"{exp_lower}_merge_residual.png")

        # 3) bland-altman 저장 경로
        save_ba = os.path.join(merge_dir, f"{exp_lower}_merge_bland_altman.png")

        # 4) cega 저장 경로
        save_cega = os.path.join(merge_dir, f"{exp_lower}_merge_cega.png")

        # 실제 그림 생성/저장
        plot_merge_actual_vs_pred(df, models, save_actual)
        plot_merge_residual(df, models, save_resid)
        plot_merge_bland_altman(df, models, save_ba)
        plot_merge_cega(df, models, save_cega)

        # 완료 로그
        print(f"✅ MERGE 저장 완료: {exp} -> {merge_dir}")


# Python 파일을 직접 실행할 때만 main()이 실행되도록 하는 표준 구조
if __name__ == "__main__":
    main()
