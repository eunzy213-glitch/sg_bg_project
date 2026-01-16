# preprocessing.py
# ============================================================
# 데이터 전처리 단계
# - 결측치 처리
# - 범주형 결측 대체
# - 잔차 기반 이상치 제거
# - Isolation Forest 기반 이상치 제거
# ============================================================

import numpy as np # 수치계산 라이브러리
import pandas as pd # DataFrame 처리 라이브러리
from sklearn.linear_model import LinearRegression # 선형회귀 모델
from sklearn.ensemble import IsolationForest  # 이상치 탐지 모델


def preprocess_and_filter_outliers(df):
    """
    입력 데이터(df)에 대해 전처리 및 이상치 제거를 수행하고,
    전처리 후 데이터(df_clean)와 제거 요약 리포트(report)를 반환합니다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임 (SG, BG 및 기타 범주형 컬럼 포함 가능)

    Returns
    -------
    df_clean : pd.DataFrame
        전처리 + 이상치 제거 후의 데이터
    report : pd.DataFrame
        전처리 단계 전/후 행(row) 수 요약
    """
        
    # 원본 보호를 위해 복사
    df = df.copy()

    # --------------------------------------------------------
    # 1️⃣ SG, BG 결측 제거
    # --------------------------------------------------------
    df = df.dropna(subset=["SG", "BG"]) # SG, BG 결측치가 있는 행 제거

    # --------------------------------------------------------
    # 2️⃣ 범주형 결측 처리
    # --------------------------------------------------------
    cat_cols = df.select_dtypes(include="object").columns # dtype이 object인 컬럼(범주형) 선택
    df[cat_cols] = df[cat_cols].fillna("Unknown") # 범주형 결측치는 "Unknown"으로 대체

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
    iso = IsolationForest( # Isolation Forest 모델 생성
        n_estimators=300,
        contamination=0.03,
        random_state=42
    )

    iso_label = iso.fit_predict(df[["SG", "BG"]]) # 이상치 탐지 수행
    mask_iso = iso_label == 1 # 정상치인 행에 대해 True

    # --------------------------------------------------------
    # 5️⃣ 최종 필터
    # --------------------------------------------------------
    final_mask = mask_residual & mask_iso # 두 조건을 모두 만족하는 행만 선택
    df_clean = df.loc[final_mask] # final_mask에 해당하는 행만 선택

    # --------------------------------------------------------
    # 6️⃣ 리포트 생성
    # --------------------------------------------------------
    report = pd.DataFrame({ # 전처리 전/후 행 개수를 간단히 요약하는 리포트
        "stage": ["original", "after_filter"],
        "rows": [len(df), len(df_clean)]
    })

    return df_clean, report # 최종 전처리 데이터와 요약 리포트 반환