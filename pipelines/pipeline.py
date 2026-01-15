# pipeline.py
# ============================================================
# SG → BG 예측 전체 파이프라인
# - 전처리
# - 모델 학습
# - 평가
# - 시각화
# - 예측 결과 CSV 저장 (Streamlit/추론용)
# - K-Fold 교차검증 시각화
# - 최종 추론 모델(pkl) 저장
# ============================================================

import os
import shutil
import pandas as pd
import joblib

from src.feature_builder import build_features
from src.preprocessing import preprocess_and_filter_outliers
from src.models import get_model_dict, train_and_predict_all
from src.evaluation import (
    evaluate_all_models_overall,
    kfold_evaluate_models
)
from src.visualization import (
    plot_scatter,
    plot_actual_vs_pred,
    plot_residual,
    plot_bland_altman,
    plot_cega,
    plot_model_metrics,
    plot_kfold_metrics
)


def run_pipeline(data_path, experiment_name, feature_mode):
    """
    하나의 실험(SG_ONLY / SG_PLUS_META)을
    처음부터 끝까지 실행하는 파이프라인
    """

    # --------------------------------------------------------
    # 0️⃣ 결과 폴더 초기화
    # --------------------------------------------------------
    results_dir = os.path.join("results", experiment_name)

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1️⃣ 데이터 로드
    # --------------------------------------------------------
    df = pd.read_csv(data_path)

    # --------------------------------------------------------
    # 2️⃣ 제외 컬럼 제거
    # --------------------------------------------------------
    drop_cols = [c for c in ["Gender", "Target_R"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --------------------------------------------------------
    # 3️⃣ 전처리 + 이상치 제거
    # --------------------------------------------------------
    df_clean, filter_report = preprocess_and_filter_outliers(df)

    # index 정합성 유지
    df_clean = df_clean.reset_index(drop=True)

    filter_report.to_csv(
        os.path.join(results_dir, "filter_report.csv"),
        index=False
    )

    # --------------------------------------------------------
    # 4️⃣ Feature 구성
    # --------------------------------------------------------
    X, y, feature_names = build_features(
        df_clean,
        mode=feature_mode
    )
    # ⚠️ X, y 는 numpy.ndarray

    # --------------------------------------------------------
    # 5️⃣ 모델 학습 및 예측 (train/test split 내부 처리)
    # --------------------------------------------------------
    models = get_model_dict()

    # pred_pack 구조:
    # {
    #   "y_test": y_test (numpy),
    #   "preds": {model_name: y_pred_array},
    #   "test_idx": test indices (df_clean 기준)
    # }
    pred_pack = train_and_predict_all(X, y, models)

    # --------------------------------------------------------
    # 6️⃣ 성능 평가 (Hold-out Test)
    # --------------------------------------------------------
    overall_metrics = evaluate_all_models_overall(pred_pack)
    overall_metrics["experiment"] = experiment_name

    overall_metrics.to_csv(
        os.path.join(results_dir, "overall_metrics.csv"),
        index=False
    )

    # --------------------------------------------------------
    # 7️⃣ 전체 데이터 분포 시각화
    # --------------------------------------------------------
    plot_scatter(df_clean, results_dir)

    y_true = pred_pack["y_test"]

    # --------------------------------------------------------
    # 8️⃣ 모델별 시각화 (모델별 폴더)
    # --------------------------------------------------------
    for model_name, y_pred in pred_pack["preds"].items():

        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        plot_actual_vs_pred(y_true, y_pred, model_name, model_dir)
        plot_residual(y_true, y_pred, model_name, model_dir)
        plot_bland_altman(y_true, y_pred, model_name, model_dir)
        plot_cega(y_true, y_pred, model_name, model_dir)

    # --------------------------------------------------------
    # 9️⃣ 모델 성능 비교 Bar Plot (R2 / RMSE / MAE / MARD)
    # --------------------------------------------------------
    plot_model_metrics(overall_metrics, results_dir)

    # --------------------------------------------------------
    # 🔟 K-Fold 교차검증 + 시각화
    # --------------------------------------------------------
    kfold_df = kfold_evaluate_models(df_clean, models)

    kfold_df.to_csv(
        os.path.join(results_dir, "kfold_metrics.csv"),
        index=False
    )

    #plot_kfold_metrics(
    #    kfold_df,
    #    results_dir
    #)

    # --------------------------------------------------------
    # 1️⃣1️⃣ 예측 결과 CSV 저장 (Streamlit / 분석용)
    # --------------------------------------------------------
    pred_rows = []
    test_idx = pred_pack["test_idx"]

    for model_name, y_pred in pred_pack["preds"].items():
        for i, idx in enumerate(test_idx):
            pred_rows.append({
                "experiment": experiment_name,
                "model": model_name,

                # SG 값은 df_clean에서 가져옴
                "SG": df_clean.loc[idx, "SG"],

                "y_true": y_true[i],
                "y_pred": y_pred[i],
                "residual": y_pred[i] - y_true[i],
            })

    pred_df = pd.DataFrame(pred_rows)

    pred_df.to_csv(
        os.path.join(results_dir, "predictions.csv"),
        index=False
    )

    print(f"✅ 실험 완료 (예측 CSV 포함): {experiment_name}")

    # --------------------------------------------------------
    # 1️⃣2️⃣ 최적 모델 저장 (추론용)
    # - SG_PLUS_META 실험에서만 수행
    # - 전체 데이터(X, y)로 재학습
    # --------------------------------------------------------
    if experiment_name == "SG_PLUS_META":

        lgbm_model = get_model_dict()["LightGBM"]

        # ⭐ build_features 결과와 동일한 feature 구성 사용
        lgbm_model.fit(X, y)

        model_save_path = os.path.join(
            "results",
            "SG_PLUS_META",
            "best_model_lightgbm.pkl"
        )

        joblib.dump(
            lgbm_model,
            model_save_path
        )

        print(f"✅ 추론용 모델 저장 완료: {model_save_path}")
