# models.py
# ============================================================
# 모델 정의 및 학습/예측 유틸
# ============================================================

import numpy as np # 수치연산 및 인덱스 배열 생성 라이브러리
from sklearn.model_selection import train_test_split # 데이터를 train/test로 나누는 함수


def get_model_dict():
    from sklearn.linear_model import LinearRegression, HuberRegressor # 선형회귀 및 Huber 회귀
    from sklearn.preprocessing import PolynomialFeatures # 다항 Feature 생성
    from sklearn.pipeline import Pipeline # 전처리+모델을 한번에 묶는 파이프라인
    from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트 회귀 모델
    from lightgbm import LGBMRegressor # LightGBM 회귀 모델

    return { # 모델들을 딕셔너리 형태로 반환
        "Linear": LinearRegression(), # 기본 선형회귀 모델

        "Polynomial": Pipeline([ # 다항 회귀 모델 (3차 다항식)
            ("poly", PolynomialFeatures(degree=3)),
            ("lr", LinearRegression())
        ]),

        "Huber": HuberRegressor(), # Huber 회귀 모델 (이상치에 강건)

        "RandomForest": RandomForestRegressor( # 랜덤포레스트 회귀 모델
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor( # LightGBM 회귀 모델
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        )
    }



def train_and_predict_all(X, y, models, test_size=0.2, random_state=42): # 모든 모델을 학습하고 test set 예측 결과를 반환

    # --------------------------------------------------------
    # train / test split
    # --------------------------------------------------------
    indices = np.arange(len(X)) # 0 ~ N-1까지 인덱스 배열 생성

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split( # train_tset_split으로 x, y, indices를 같은 방식으로 나눔
        X,
        y,
        indices,
        test_size=test_size,
        random_state=random_state
    )

    preds = {} # 모델별 예측 결과를 저장할 딕셔너리

    # --------------------------------------------------------
    # 모델별 학습 및 예측
    # --------------------------------------------------------
    for name, model in models.items(): # dict의 모델명, 모델객체 순회
        model.fit(X_train, y_train) # 모델 학습
        preds[name] = model.predict(X_test) # 테스트셋 예측값 저장

    # --------------------------------------------------------
    # 반환 구조
    # --------------------------------------------------------
    pred_pack = {
        "y_test": y_test, # 테스트 타겟
        "preds": preds, # 모델별 예측값 딕셔너리
        "test_idx": idx_test   # 원본 df 기준의 test 행 인덱스
    }

    return pred_pack # 학습/예측 패키지 반환
