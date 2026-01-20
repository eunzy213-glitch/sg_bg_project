# models.py
# ============================================================
# 모델 정의 및 학습/예측 유틸
# ============================================================

import numpy as np # 수치연산 및 인덱스 배열 생성 라이브러리
import logging     # ✅ 로그 출력용 라이브러리 (추가)
from sklearn.model_selection import train_test_split # 데이터를 train/test로 나누는 함수

# ------------------------------------------------------------
# ✅ Logger 설정 (추가)
# ------------------------------------------------------------
logger = logging.getLogger(__name__)


def get_model_dict():
    from sklearn.linear_model import LinearRegression, HuberRegressor # 선형회귀 및 Huber 회귀
    from sklearn.preprocessing import PolynomialFeatures # 다항 Feature 생성
    from sklearn.pipeline import Pipeline # 전처리+모델을 한번에 묶는 파이프라인
    from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트 회귀 모델
    from lightgbm import LGBMRegressor # LightGBM 회귀 모델

    logger.info("🔹 모델 딕셔너리 생성 시작")

    model_dict = { # 모델들을 딕셔너리 형태로 반환
        "Linear": LinearRegression(), # 기본 선형회귀 모델

        "Polynomial": Pipeline([ # 다항 회귀 모델 (3차 다항식)
            ("poly", PolynomialFeatures(degree=3)),
            ("lr", LinearRegression())
        ]),

        "Huber": HuberRegressor(), # Huber 회귀 모델 (이상치에 강건)

        "RandomForest": RandomForestRegressor( # 랜덤포레스트 회귀 모델
            n_estimators=300,
            random_state=42,
            n_jobs=1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
            subsample=1.0,
            colsample_bytree=1.0,
            deterministic=True,
            force_row_wise=True
        )
    }

    logger.info(f"🔹 사용 모델 목록: {list(model_dict.keys())}")

    return model_dict



def train_and_predict_all(X, y, models, test_size=0.2, random_state=42): # 모든 모델을 학습하고 test set 예측 결과를 반환

    logger.info("🔹 모델 학습 및 예측 시작")
    logger.info(f"🔹 입력 데이터 shape: X={X.shape}, y={y.shape}")

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

    logger.info(
        f"🔹 Train/Test split 완료 | "
        f"Train={X_train.shape[0]}, Test={X_test.shape[0]}"
    )

    preds = {} # 모델별 예측 결과를 저장할 딕셔너리

    # --------------------------------------------------------
    # 모델별 학습 및 예측
    # --------------------------------------------------------
    for name, model in models.items(): # dict의 모델명, 모델객체 순회
        logger.info(f"🔹 모델 학습 시작: {name}")

        model.fit(X_train, y_train) # 모델 학습
        preds[name] = model.predict(X_test) # 테스트셋 예측값 저장

        logger.info(
            f"🔹 모델 예측 완료: {name} | "
            f"예측 샘플 수={len(preds[name])}"
        )

    # --------------------------------------------------------
    # 반환 구조
    # --------------------------------------------------------
    pred_pack = {
        "y_test": y_test, # 테스트 타겟
        "preds": preds, # 모델별 예측값 딕셔너리
        "test_idx": idx_test   # 원본 df 기준의 test 행 인덱스
    }

    logger.info("🔹 모든 모델 학습/예측 완료")

    return pred_pack # 학습/예측 패키지 반환
