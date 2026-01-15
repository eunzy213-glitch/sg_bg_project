# pipelines/explain_pipeline.py
# ============================================================
# SHAP / LIME Explainability 전용 파이프라인
#
# 🔑 핵심 설계
# - 학습/추론 파이프라인과 분리
# - Explain 전용 One-Hot Encoding 사용
# - 서브 카테고리 단위 SHAP / LIME 해석 가능
#
# 결과 저장 구조:
# results/
# └── SG_PLUS_META/
#     ├── EXPLAIN_LightGBM/
#     └── EXPLAIN_RandomForest/
# ============================================================

import os
import pandas as pd
import numpy as np

from src.preprocessing import preprocess_and_filter_outliers
from src.models import get_model_dict
from src.explainability import run_shap_analysis, run_lime_analysis


# ============================================================
# ⭐ 프로젝트 루트 경로 자동 계산
# 이 파일 위치: sg_bg_project/pipelines/explain_pipeline.py
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR == sg_bg_project


# ============================================================
# ⭐ Explain 전용 One-Hot Feature 생성 함수
# ============================================================
CATEGORICAL_COLS = [
    "Meal_Status",
    "BMI_Class",
    "Age_Group",
    "Exercise",
    "Family_History",
    "Pregnancy",
]


def build_explain_features(df: pd.DataFrame):
    """
    SHAP / LIME 전용 feature 생성

    - SG: 수치형 그대로 사용
    - 범주형 변수: One-Hot Encoding
    - 학습/추론 파이프라인과 분리된 Explain 전용 설계
    """

    # --------------------------------------------------------
    # 1️⃣ 수치형 Feature
    # --------------------------------------------------------
    X_num = df[["SG"]]

    # --------------------------------------------------------
    # 2️⃣ 범주형 Feature → One-Hot Encoding
    # --------------------------------------------------------
    X_cat = pd.get_dummies(
        df[CATEGORICAL_COLS],
        prefix=CATEGORICAL_COLS
    )

    # --------------------------------------------------------
    # 3️⃣ 결합
    # --------------------------------------------------------
    X_explain = pd.concat([X_num, X_cat], axis=1)

    feature_names = X_explain.columns.tolist()

    return X_explain.values, feature_names


def run_explain_pipeline(
    data_path: str,
    experiment_name: str,
    target_models: list | None = None
):
    """
    SHAP / LIME 설명가능성 분석 파이프라인

    Parameters
    ----------
    data_path : str
        원본 데이터 경로 (프로젝트 루트 기준)
        예: "data/dataset.csv"
    experiment_name : str
        "SG_ONLY" or "SG_PLUS_META"
    target_models : list or None
        설명할 모델 이름 리스트
        예) ["LightGBM", "RandomForest"]
        None이면 모든 모델 수행
    """

    # --------------------------------------------------------
    # 1️⃣ 데이터 로드 (실행 위치 무관)
    # --------------------------------------------------------
    data_path = os.path.join(BASE_DIR, data_path)
    df = pd.read_csv(data_path)

    # --------------------------------------------------------
    # 2️⃣ 불필요 컬럼 제거
    # --------------------------------------------------------
    drop_cols = [c for c in ["Gender", "Target_R"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --------------------------------------------------------
    # 3️⃣ 전처리 + 이상치 제거
    # --------------------------------------------------------
    df_clean, _ = preprocess_and_filter_outliers(df)

    # index 정합성 유지
    df_clean = df_clean.reset_index(drop=True)

    # --------------------------------------------------------
    # 4️⃣ Explain 전용 Feature 구성 (⭐ One-Hot)
    # --------------------------------------------------------
    # 타깃
    y = df_clean["BG"].values

    # Explain 전용 Feature
    X_explain, feature_names = build_explain_features(df_clean)

    # --------------------------------------------------------
    # 5️⃣ 모델 불러오기
    # --------------------------------------------------------
    models = get_model_dict()

    if target_models is not None:
        models = {
            name: model
            for name, model in models.items()
            if name in target_models
        }

    # --------------------------------------------------------
    # 6️⃣ Explain 결과 저장 루트
    # --------------------------------------------------------
    base_results_dir = os.path.join(
        BASE_DIR,
        "results",
        experiment_name
    )
    os.makedirs(base_results_dir, exist_ok=True)

    # --------------------------------------------------------
    # 7️⃣ 모델별 SHAP / LIME 수행
    # --------------------------------------------------------
    for model_name, model in models.items():

        print(f"🔍 Explain 시작: {model_name}")

        explain_dir = os.path.join(
            base_results_dir,
            f"EXPLAIN_{model_name}"
        )
        os.makedirs(explain_dir, exist_ok=True)

        # ----------------------------------------------------
        # 모델 학습 (Explain 전용 feature 사용)
        # ----------------------------------------------------
        model.fit(X_explain, y)

        # ----------------------------------------------------
        # SHAP 분석
        # ----------------------------------------------------
        run_shap_analysis(
            model=model,
            X_train=X_explain,
            X_test=X_explain,
            feature_names=feature_names,
            save_dir=explain_dir
        )

        # ----------------------------------------------------
        # LIME 분석
        # ----------------------------------------------------
        run_lime_analysis(
            model=model,
            X_train=X_explain,
            X_test=X_explain,
            feature_names=feature_names,
            save_dir=explain_dir
        )

        print(f"✅ SHAP/LIME 완료: {model_name}")

    print(f"\n🎉 Explain pipeline 완료: {experiment_name}")


# ------------------------------------------------------------
# 단독 실행용
# ------------------------------------------------------------
if __name__ == "__main__":

    run_explain_pipeline(
        data_path="data/dataset.csv",   # ⭐ 프로젝트 루트 기준
        experiment_name="SG_PLUS_META",
        target_models=["LightGBM", "RandomForest"]
    )
