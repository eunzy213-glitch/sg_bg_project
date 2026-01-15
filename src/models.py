# models.py
# ============================================================
# 모델 정의 및 학습/예측 유틸
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split


def get_model_dict():
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor

    return {
        "Linear": LinearRegression(),

        "Polynomial": Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("lr", LinearRegression())
        ]),

        "Huber": HuberRegressor(),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        )
    }



def train_and_predict_all(X, y, models, test_size=0.2, random_state=42):
    """
    모든 모델을 학습하고 test set 예측 결과를 반환

    Returns
    -------
    pred_pack : dict
        {
            "y_test": y_test,
            "preds": {model_name: y_pred},
            "test_idx": test indices (원본 df 기준)
        }
    """

    # --------------------------------------------------------
    # train / test split
    # --------------------------------------------------------
    indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=random_state
    )

    preds = {}

    # --------------------------------------------------------
    # 모델별 학습 및 예측
    # --------------------------------------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)

    # --------------------------------------------------------
    # 반환 구조 (⭐ test_idx 포함)
    # --------------------------------------------------------
    pred_pack = {
        "y_test": y_test,
        "preds": preds,
        "test_idx": idx_test   # 👈 핵심
    }

    return pred_pack
