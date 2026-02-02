"""
SEG 분석이 통합된 파이프라인 실행 예제

기존 pipeline.py를 확장하여 SEG 분석을 추가하는 예제 코드
이 코드는 main.py에서 호출되거나 독립 실행 가능
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List

# 기존 모듈 임포트 (프로젝트 구조에 맞게 조정)
# from src.preprocessing import preprocess_data
# from src.feature_builder import build_features
# from src.models import train_model
# from src.visualization import create_visualizations

# SEG 분석 모듈 임포트
from src.evaluation_with_seg import (
    evaluate_model_with_seg,
    create_combined_summary,
    compare_seg_across_models
)

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline_with_seg(
    data_path: str,
    results_base_dir: str = "results",
    experiment_name: str = "SG_PLUS_META",
    models_to_train: List[str] = None
) -> Dict[str, Any]:
    """
    SEG 분석이 포함된 전체 파이프라인 실행
    
    Args:
        data_path: 데이터 파일 경로
        results_base_dir: 결과 저장 기본 디렉토리
        experiment_name: 실험 이름 (SG_ONLY 또는 SG_PLUS_META)
        models_to_train: 학습할 모델 목록
        
    Returns:
        Dict: 전체 평가 결과
    """
    logger.info("="*80)
    logger.info(f"Starting Pipeline with SEG Analysis - Experiment: {experiment_name}")
    logger.info("="*80)
    
    # 기본 설정
    if models_to_train is None:
        models_to_train = [
            'Linear',
            'Polynomial',
            'Huber',
            'RandomForest',
            'LightGBM'
        ]
    
    # 결과 디렉토리 설정
    results_dir = Path(results_base_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {results_dir}")
    
    # ========================================================================
    # 1단계: 데이터 로드 및 전처리
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("="*80)
    
    # 실제 프로젝트에서는 다음과 같이 사용:
    # df = pd.read_csv(data_path)
    # X_train, X_test, y_train, y_test = preprocess_data(df, experiment_name)
    
    # 예제를 위한 더미 데이터 생성 (실제로는 위 코드로 대체)
    np.random.seed(42)
    n_samples = 1000
    
    # 실제 BG 값 생성 (70-300 mg/dL 범위)
    y_test_example = np.random.uniform(70, 300, n_samples)
    
    # 예측값 생성 (약간의 노이즈 추가)
    y_pred_example = y_test_example + np.random.normal(0, 20, n_samples)
    y_pred_example = np.clip(y_pred_example, 50, 350)  # 범위 제한
    
    logger.info(f"Data loaded: {n_samples} test samples")
    
    # ========================================================================
    # 2단계: 모델 학습 (기존 파이프라인)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Model Training")
    logger.info("="*80)
    
    # 실제 프로젝트에서는:
    # trained_models = {}
    # for model_name in models_to_train:
    #     trained_models[model_name] = train_model(model_name, X_train, y_train)
    #     logger.info(f"Model trained: {model_name}")
    
    logger.info(f"Models to train: {', '.join(models_to_train)}")
    logger.info("(In actual pipeline, models would be trained here)")
    
    # ========================================================================
    # 3단계: 모델 평가 (기본 지표 + SEG)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Model Evaluation with SEG Analysis")
    logger.info("="*80)
    
    all_evaluation_results = {}
    
    for model_name in models_to_train:
        logger.info(f"\n--- Evaluating {model_name} ---")
        
        # 실제 프로젝트에서는:
        # y_pred = trained_models[model_name].predict(X_test)
        
        # 예제를 위해 약간 다른 예측값 생성
        noise_level = np.random.uniform(15, 30)
        y_pred = y_test_example + np.random.normal(0, noise_level, n_samples)
        y_pred = np.clip(y_pred, 50, 350)
        
        # 평가 수행 (기본 지표 + SEG)
        evaluation_results = evaluate_model_with_seg(
            y_true=y_test_example,
            y_pred=y_pred,
            model_name=model_name,
            results_dir=results_dir,
            experiment_name=experiment_name,
            save_detailed_results=True
        )
        
        all_evaluation_results[model_name] = evaluation_results
    
    # ========================================================================
    # 4단계: 통합 결과 요약
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Creating Combined Summary")
    logger.info("="*80)
    
    # 통합 요약 테이블 생성
    summary_path = results_dir / "combined_summary_with_seg.csv"
    summary_df = create_combined_summary(
        all_results=all_evaluation_results,
        save_path=summary_path
    )
    
    # SEG 비교 시각화 생성
    seg_comparison_path = results_dir / "seg_comparison_all_models.png"
    compare_seg_across_models(
        all_results=all_evaluation_results,
        save_path=seg_comparison_path,
        experiment_name=experiment_name
    )
    
    # ========================================================================
    # 5단계: 최종 요약 출력
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Final Summary")
    logger.info("="*80)
    
    # 최고 성능 모델 찾기 (Clinically-Acceptable 기준)
    best_model = None
    best_acceptable_rate = 0.0
    
    for model_name, results in all_evaluation_results.items():
        acceptable_rate = results['seg_results']['statistics']['Clinically-Acceptable']
        if acceptable_rate > best_acceptable_rate:
            best_acceptable_rate = acceptable_rate
            best_model = model_name
    
    logger.info(f"\nBest Model (by SEG Clinically-Acceptable): {best_model}")
    logger.info(f"  - Clinically-Acceptable Rate: {best_acceptable_rate:.2f}%")
    
    best_metrics = all_evaluation_results[best_model]['metrics']
    logger.info(f"  - R²: {best_metrics['R2']:.4f}")
    logger.info(f"  - MARD: {best_metrics['MARD']:.2f}%")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline with SEG Analysis Completed Successfully!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("Generated files:")
    logger.info(f"  - SEG plots for each model: seg_<model_name>.png")
    logger.info(f"  - SEG summaries: seg_<model_name>_summary.csv")
    logger.info(f"  - Detailed results: detailed_results_<model_name>.csv")
    logger.info(f"  - Combined summary: combined_summary_with_seg.csv")
    logger.info(f"  - SEG comparison: seg_comparison_all_models.png")
    
    return all_evaluation_results


def main():
    """
    메인 실행 함수 - 독립 실행 또는 main.py에서 import하여 사용
    """
    # 설정
    DATA_PATH = "data/synthetic_data.csv"  # 실제 데이터 경로로 변경
    RESULTS_DIR = "results"
    
    # 두 가지 실험 모두 실행
    experiments = ["SG_ONLY", "SG_PLUS_META"]
    
    all_experiment_results = {}
    
    for experiment_name in experiments:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Experiment: {experiment_name}")
        logger.info(f"{'='*80}\n")
        
        results = run_pipeline_with_seg(
            data_path=DATA_PATH,
            results_base_dir=RESULTS_DIR,
            experiment_name=experiment_name
        )
        
        all_experiment_results[experiment_name] = results
    
    # 실험 간 비교
    logger.info("\n" + "="*80)
    logger.info("Cross-Experiment Comparison")
    logger.info("="*80)
    
    for exp_name, exp_results in all_experiment_results.items():
        logger.info(f"\n{exp_name}:")
        for model_name, results in exp_results.items():
            seg_stats = results['seg_results']['statistics']
            logger.info(f"  {model_name}: "
                       f"Acceptable={seg_stats['Clinically-Acceptable']:.2f}%, "
                       f"MARD={results['metrics']['MARD']:.2f}%")


if __name__ == "__main__":
    main()