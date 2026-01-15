# preprocessing.py
# ============================================================
# 데이터 품질을 보장하기 위한 전처리 단계
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


def preprocess_and_filter_outliers(df):
    # 원본 보호를 위해 복사
    df = df.copy()

    # --------------------------------------------------------
    # 1️⃣ SG, BG 결측 제거
    # --------------------------------------------------------
    df = df.dropna(subset=["SG", "BG"])

    # --------------------------------------------------------
    # 2️⃣ 범주형 결측 처리
    # --------------------------------------------------------
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # --------------------------------------------------------
    # 3️⃣ 잔차 기반 이상치 제거
    # --------------------------------------------------------
    X = df[["SG"]].values
    y = df["BG"].values

    # SG → BG 선형 회귀 (baseline)
    lr = LinearRegression()
    lr.fit(X, y)

    # 예측 및 잔차 계산
    residual = np.abs(y - lr.predict(X))

    # 평균 + 3표준편차 기준
    threshold = residual.mean() + 3 * residual.std()
    mask_residual = residual <= threshold

    # --------------------------------------------------------
    # 4️⃣ Isolation Forest
    # --------------------------------------------------------
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42
    )

    iso_label = iso.fit_predict(df[["SG", "BG"]])
    mask_iso = iso_label == 1

    # --------------------------------------------------------
    # 5️⃣ 최종 필터
    # --------------------------------------------------------
    final_mask = mask_residual & mask_iso
    df_clean = df.loc[final_mask]

    # --------------------------------------------------------
    # 6️⃣ 리포트 생성
    # --------------------------------------------------------
    report = pd.DataFrame({
        "stage": ["original", "after_filter"],
        "rows": [len(df), len(df_clean)]
    })

    return df_clean, report