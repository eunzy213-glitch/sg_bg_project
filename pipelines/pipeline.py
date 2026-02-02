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

import os # 파일/폴더 경로 라이브러리
import shutil # 폴더 시스템 작업 라이브러리
import pandas as pd # DataFrame 처리 라이브러리
import joblib # 학습된 모델 .pkl 형태로 저장/로드하기 위한 라이브러리
import logging # ✅ 로그 출력용 라이브러리 (추가)
import numpy as np # 수치연산 라이브러리
from pathlib import Path


from src.feature_builder import build_features # 모델에 넣을 x, y, feature_name 생성 
from src.preprocessing import preprocess_and_filter_outliers # 전처리 전체 로직 처리
from src.models import get_model_dict, train_and_predict_all  # 모델들을 dict 형태로 반환, 학습 및 예측 수행 
from src.evaluation import (evaluate_all_models_overall, kfold_evaluate_models) # 모델별 성능지표 반환, k-fold 교차검증 수행
from src.visualization import (plot_scatter, plot_actual_vs_pred, plot_residual, plot_bland_altman, plot_cega, plot_model_metrics) # 시각화 함수들
from src.evaluation_with_seg import (evaluate_model_with_seg, create_combined_summary, compare_seg_across_models) # ✅ SEG 분석 관련 함수들 (추가)

# --------------------------------------------------------
# ✅ Logger 설정 (추가)
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path, experiment_name, feature_mode):
    """
    하나의 실험(SG_ONLY / SG_PLUS_META)을
    처음부터 끝까지 실행하는 파이프라인
    """

    logger.info(f"🚀 Pipeline 시작 | experiment={experiment_name}, feature_mode={feature_mode}")

    try:  # ✅ 전체 파이프라인 보호 (추가)

        # --------------------------------------------------------
        # 0️⃣ 결과 폴더 초기화
        # --------------------------------------------------------
        results_dir = os.path.join("results", experiment_name) # 결과를 저장할 폴더 생성

        if os.path.exists(results_dir): # 재실행 시 기존 결과 삭제
            logger.warning(f"⚠️ 기존 결과 폴더 삭제: {results_dir}")
            try:  # ✅ PermissionError 방어 (추가)
                shutil.rmtree(results_dir)
            except PermissionError as e:
                logger.error(f"❌ 결과 폴더 삭제 실패 (권한 문제): {e}")
                raise e

        os.makedirs(results_dir, exist_ok=True) # 폴더 생성
        logger.info(f"📁 결과 폴더 생성 완료: {results_dir}")

        # --------------------------------------------------------
        # 1️⃣ 데이터 로드
        # --------------------------------------------------------
        logger.info(f"📂 데이터 로드 시작: {data_path}")
        df = pd.read_csv(data_path) # 원본데이터 로드
        logger.info(f"✅ 데이터 로드 완료 | rows={len(df)}, cols={len(df.columns)}")

        # --------------------------------------------------------
        # 2️⃣ 제외 컬럼 제거
        # --------------------------------------------------------
        drop_cols = [c for c in ["Gender", "Target_R"] if c in df.columns] # Gender, Target_R 컬럼 제거
        df = df.drop(columns=drop_cols) # drop_cols에 들어있는 컬럼 제거
        logger.info(f"🧹 제거된 컬럼: {drop_cols}")

        # --------------------------------------------------------
        # 3️⃣ 전처리 + 이상치 제거
        # --------------------------------------------------------
        logger.info("🧪 전처리 및 이상치 제거 시작")
        df_clean, filter_report = preprocess_and_filter_outliers(df) # 전처리 및 이상치 제거 수행

        # index 정합성 유지
        df_clean = df_clean.reset_index(drop=True) # 인덱스 재정렬

        filter_report.to_csv( # 전처리/이상치 제거 리포트 csv 저장
            os.path.join(results_dir, "filter_report.csv"),
            index=False
        )

        logger.info(f"✅ 전처리 완료 | before={len(df)}, after={len(df_clean)}")

        # --------------------------------------------------------
        # 4️⃣ Feature 구성 (모델에 넣을 X, y 생성)
        # --------------------------------------------------------
        logger.info("🧩 Feature 구성 시작")
        X, y, feature_names = build_features(
            df_clean,
            mode=feature_mode
        )
        logger.info(f"✅ Feature 구성 완료 | X.shape={X.shape}, feature_count={len(feature_names)}")

        # --------------------------------------------------------
        # 5️⃣ 모델 학습 및 예측 (train/test split 내부 처리)
        # --------------------------------------------------------
        models = get_model_dict() # 사용할 모델을 dict 형태로 받아옴
        logger.info(f"🤖 사용 모델 목록: {list(models.keys())}")

        pred_pack = train_and_predict_all(X, y, models) # test set에 대한 예측값 반환
        logger.info("✅ 모델 학습 및 예측 완료")

        # ========================================================
        # 🔥 5️⃣-1️⃣ Weighted Ensemble (Hold-out Test 정식 편입)
        # ========================================================
        if experiment_name == "SG_PLUS_META":

            logger.info("🧩 Weighted Ensemble Hold-out 예측 생성 시작")

            ensemble_weights = {
                "Linear": 0.05,
                "Polynomial": 0.10,
                "Huber": 0.15,
                "RandomForest": 0.20,
                "LightGBM": 0.50
            }

            y_true = pred_pack["y_test"]
            ensemble_pred = np.zeros_like(y_true, dtype=float)

            for model_name, weight in ensemble_weights.items():
                ensemble_pred += weight * pred_pack["preds"][model_name]

            # ✅ 정식 모델로 등록
            pred_pack["preds"]["WeightedEnsemble"] = ensemble_pred

            logger.info("✅ Weighted Ensemble Hold-out 예측 완료")

        # --------------------------------------------------------
        # 6️⃣ 성능 평가 (Hold-out Test)
        # --------------------------------------------------------
        logger.info("📊 모델 성능 평가 시작")
        overall_metrics = evaluate_all_models_overall(pred_pack) # 모델별 성능지표 계산
        overall_metrics["experiment"] = experiment_name # 실험명 컬럼 추가

        overall_metrics.to_csv(
            os.path.join(results_dir, "overall_metrics.csv"),
            index=False
        )

        logger.info("✅ 모델 성능 지표 저장 완료")
        
        # --------------------------------------------------------
        # ✅ SEG 기반 추가 평가 (Step2-2 반영)
        # --------------------------------------------------------
        logger.info("📌 SEG 기반 평가 시작 (SEG Analysis)")

        all_results = {}  # ✅ 모델별 SEG 결과 저장용 dict

        for model_name, y_pred in pred_pack["preds"].items():

            logger.info(f"   ▶ SEG 평가 수행 중: {model_name}")

            # ✅ 반드시 먼저 model_dir 생성해야 함
            model_dir = os.path.join(results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # ✅ SEG 포함 평가 실행 (Path로 변환해서 전달)
            results = evaluate_model_with_seg(
                y_true=pred_pack["y_test"],
                y_pred=y_pred,
                model_name=model_name,
                results_dir=Path(model_dir),        # ✅ Path 변환 유지
                experiment_name=experiment_name
            )

            # ✅ 모델별 결과 저장
            all_results[model_name] = results

            # --------------------------------------------------------
            # ✅ 로그 출력 (KeyError 방지 수정 완료)
            # --------------------------------------------------------
            metrics = results["metrics"]
            seg_stats = results["seg_results"]["statistics"]

            # ✅ SEG 통계 key 확인 출력
            logger.info(f"      📌 SEG statistics keys: {list(seg_stats.keys())}")

            # ✅ Acceptable_% key가 없을 경우 안전 처리
            acceptable = seg_stats.get("Acceptable_%", None)

            if acceptable is not None:
                logger.info(
                    f"      ✅ {model_name}: "
                    f"MARD={metrics['MARD']:.2f}%, "
                    f"SEG Acceptable={acceptable:.2f}%"
                )
            else:
                logger.info(
                    f"      ✅ {model_name}: "
                    f"MARD={metrics['MARD']:.2f}%, "
                    f"SEG Acceptable key not found"
                )

        # --------------------------------------------------------
        # ✅ 평가 루프 종료 후 요약 생성 (Step2-3 반영)
        # --------------------------------------------------------
        logger.info("📌 SEG 통합 요약 파일 생성 시작")

        # ✅ 통합 요약 CSV 생성 (experiment 폴더 루트에 저장)
        summary_path = Path(results_dir) / "combined_summary_with_seg.csv"
        create_combined_summary(all_results, summary_path)

        # ✅ SEG 모델 비교 Plot 생성 (experiment 폴더 루트에 저장)
        seg_comparison_path = Path(results_dir) / "seg_comparison_all_models.png"
        compare_seg_across_models(
            all_results,
            seg_comparison_path,
            experiment_name
        )

        logger.info("✅ SEG 분석 완료 (모델별 폴더에 저장됨)")
        
        
        # --------------------------------------------------------
        # 7️⃣ 전체 데이터 분포 시각화
        # --------------------------------------------------------
        logger.info("📈 SG vs BG 전체 산점도 시각화")
        plot_scatter(df_clean, results_dir)

        y_true = pred_pack["y_test"]

        # --------------------------------------------------------
        # 8️⃣ 모델별 시각화 (모델별 폴더)
        # --------------------------------------------------------
        logger.info("🖼️ 모델별 시각화 생성 시작")

        for model_name, y_pred in pred_pack["preds"].items():
            logger.info(f"   ▶ 시각화 생성 중: {model_name}")

            model_dir = os.path.join(results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            plot_actual_vs_pred(y_true, y_pred, model_name, model_dir)
            plot_residual(y_true, y_pred, model_name, model_dir)
            plot_bland_altman(y_true, y_pred, model_name, model_dir)
            plot_cega(y_true, y_pred, model_name, model_dir)

        # --------------------------------------------------------
        # 9️⃣ 모델 성능 비교 Bar Plot
        # --------------------------------------------------------
        logger.info("📊 모델 성능 비교 Bar Plot 생성")
        plot_model_metrics(overall_metrics, results_dir)

        # --------------------------------------------------------
        # 🔟 K-Fold 교차검증 + 시각화
        # --------------------------------------------------------
        logger.info("🔁 K-Fold 교차검증 시작")
        kfold_df = kfold_evaluate_models(df_clean, models)

        kfold_df.to_csv(
            os.path.join(results_dir, "kfold_metrics.csv"),
            index=False
        )

        logger.info("✅ K-Fold 결과 저장 완료")

        # --------------------------------------------------------
        # 1️⃣1️⃣ 예측 결과 CSV 저장 (Streamlit / 분석용)
        # --------------------------------------------------------
        logger.info("💾 예측 결과 CSV 생성 시작")

        pred_rows = []
        test_idx = pred_pack["test_idx"]

        for model_name, y_pred in pred_pack["preds"].items():
            for i, idx in enumerate(test_idx):
                pred_rows.append({
                    "experiment": experiment_name,
                    "model": model_name,
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

        logger.info("✅ 예측 결과 CSV 저장 완료")

        # --------------------------------------------------------
        # 1️⃣2️⃣ 최적 모델 저장 (추론용)
        # --------------------------------------------------------
        if experiment_name == "SG_PLUS_META":

            logger.info("🏆 최종 LightGBM 모델 전체 데이터로 재학습")

            lgbm_model = get_model_dict()["LightGBM"]
            lgbm_model.fit(X, y)

            model_save_path = os.path.join(
                "results",
                "SG_PLUS_META",
                "best_model_lightgbm.pkl"
            )

            joblib.dump(lgbm_model, model_save_path)

            logger.info(f"✅ 추론용 모델 저장 완료: {model_save_path}")

        logger.info(f"🎉 Pipeline 종료: {experiment_name}")

    except Exception as e:
        logger.exception("🔥 Pipeline 실행 중 치명적 오류 발생")
        raise e
