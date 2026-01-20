# evaluation.py
# ============================================================
# 이 파일은 "모델이 얼마나 잘 예측했는가?"를
# 통계적 + 임상적 관점에서 평가하는 모든 로직을 담고 있습니다.
#
# 포함 내용:
# - 회귀 성능 지표 (R2, RMSE, MAE)
# - 임상 핵심 지표 (MARD)
# - Clarke Error Grid Analysis (CEGA)
# - Bland–Altman 통계
# - K-Fold 교차검증
# ============================================================

# 수치 계산을 위한 라이브러리
import numpy as np
import pandas as pd

# sklearn의 회귀 성능 지표 함수들
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import KFold # K-Fold 교차검증용
from sklearn.base import clone


# ------------------------------------------------------------
# RMSE 계산 함수
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    """
    RMSE = sqrt(mean((y_true - y_pred)^2))
    → 오차 크기를 mg/dL 단위로 직관적으로 표현
    """
    return np.sqrt(mean_squared_error(y_true, y_pred)) # np.sprt를 씌워 RMSE로 변환


# ------------------------------------------------------------
# MARD 계산 함수
# ------------------------------------------------------------
def mard(y_true, y_pred, eps=1e-6):
    """
    MARD (Mean Absolute Relative Difference)
    = mean(|pred - true| / true) * 100

    혈당 센서/추정 모델에서 매우 중요한 임상 지표
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + eps)) * 100


# ------------------------------------------------------------
# Clarke Error Grid Analysis (CEGA)
# ------------------------------------------------------------
def clarke_error_grid(y_true, y_pred):
    """
    각 예측을 Zone A~E로 분류
    Zone A/B 비율이 임상적으로 중요
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    zones = np.array(["A"] * len(y_true), dtype=object)

    ratio = y_pred / np.clip(y_true, 1e-6, None)

    # Zone A (임상적으로 정확)
    zone_a = (
        ((y_true >= 70) & (np.abs(ratio - 1) <= 0.2)) |
        ((y_true < 70) & (np.abs(y_pred - y_true) <= 20))
    )
    zones[zone_a] = "A"

    # Zone E (완전히 반대 판단 → 위험)
    zone_e = (
        ((y_true < 70) & (y_pred > 180)) |
        ((y_true > 180) & (y_pred < 70))
    )
    zones[zone_e] = "E"

    # Zone D (치료 누락 위험)
    zone_d = (
        ((y_true > 240) & (70 <= y_pred) & (y_pred <= 180)) |
        ((y_true < 70) & (70 <= y_pred) & (y_pred <= 180))
    )
    zones[zone_d] = "D"

    # Zone C (불필요한 치료 위험)
    zone_c = (
        ((70 <= y_true) & (y_true <= 180)) &
        ((y_pred < 70) | (y_pred > 240))
    )
    zones[zone_c] = "C"

    # 나머지는 Zone B
    zones[~(zone_a | zone_c | zone_d | zone_e)] = "B"

    return zones


# ------------------------------------------------------------
# CEGA 요약 통계
# ------------------------------------------------------------
def cega_summary(y_true, y_pred):
    zones = clarke_error_grid(y_true, y_pred) # 각 샘플의 zone 계산

    return { # Zone A~E 비율 계산
        f"CEGA_{z}": np.mean(zones == z) * 100
        for z in ["A", "B", "C", "D", "E"]
    }


# ------------------------------------------------------------
# Bland–Altman 통계
# ------------------------------------------------------------
def bland_altman(y_true, y_pred):
    diff = y_pred - y_true # 예측값 - 실제값
    mean_diff = np.mean(diff) # 평균 차이
    std_diff = np.std(diff, ddof=1) # 표준편차

    return {
        "BA_bias": mean_diff,
        "BA_LoA_low": mean_diff - 1.96 * std_diff,
        "BA_LoA_high": mean_diff + 1.96 * std_diff
    }


# ------------------------------------------------------------
# 단일 모델 평가
# ------------------------------------------------------------
def evaluate_single(y_true, y_pred): # 하나의 모델에 대해 모든 평가지표를 한번에 계산
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mard": mard(y_true, y_pred)
    }

    metrics.update(cega_summary(y_true, y_pred)) # cega 요약 통계 추가
    metrics.update(bland_altman(y_true, y_pred)) # bland-altman 통계 추가

    return metrics


# ------------------------------------------------------------
# 전체 모델 평가
# ------------------------------------------------------------
def evaluate_all_models_overall(pred_pack):
    y_test = pred_pack["y_test"]
    preds = pred_pack["preds"]

    rows = []

    for model_name, y_pred in preds.items():
        result = evaluate_single(y_test, y_pred)
        result["model"] = model_name
        rows.append(result)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# K-Fold 교차검증
# ------------------------------------------------------------
def kfold_evaluate_models(df, models, n_splits=5):
    X = df[["SG"]].values
    y = df["BG"].values

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42   # ✅ 재현성 고정
    )

    rows = []

    for model_name, model in models.items():
        for fold, (tr, te) in enumerate(kf.split(X), start=1):

            # ------------------------------------------------
            # ✅ Pipeline / 모든 모델에 안전한 clone
            # ------------------------------------------------
            model_clone = clone(model)

            model_clone.fit(X[tr], y[tr])
            pred = model_clone.predict(X[te])

            result = evaluate_single(y[te], pred)
            result["model"] = model_name
            result["fold"] = fold

            rows.append(result)

    return pd.DataFrame(rows)
