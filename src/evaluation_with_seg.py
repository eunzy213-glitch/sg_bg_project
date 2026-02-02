"""
평가 지표 계산 모듈 (SEG 포함)

기존 evaluation.py의 기능에 SEG 분석을 추가한 확장 버전
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# SEG 분석 모듈 임포트
from src.seg_analysis import SurveillanceErrorGrid, evaluate_seg_for_model

# 로거 설정
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    기본 평가 지표 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        Dict: 평가 지표 딕셔너리 (R2, RMSE, MAE, MARD)
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MARD (Mean Absolute Relative Difference) - 임상적 상대 오차
    # MARD = mean(|predicted - actual| / actual) * 100
    mard = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MARD': mard
    }


def evaluate_model_with_seg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    results_dir: Path,
    experiment_name: str = "",
    save_detailed_results: bool = True
) -> Dict[str, Any]:
    """
    모델 평가 수행 (기본 지표 + SEG 분석)
    
    Args:
        y_true: 실제 BG 값
        y_pred: 예측 BG 값
        model_name: 모델 이름
        results_dir: 결과 저장 디렉토리
        experiment_name: 실험 이름
        save_detailed_results: 상세 결과 저장 여부
        
    Returns:
        Dict: 평가 결과 (metrics, seg_results)
    """
    logger.info(f"Evaluating model: {model_name}")
    
    # 1. 기본 평가 지표 계산
    metrics = calculate_metrics(y_true, y_pred)
    
    logger.info(f"Basic Metrics for {model_name}:")
    logger.info(f"  - R²: {metrics['R2']:.4f}")
    logger.info(f"  - RMSE: {metrics['RMSE']:.2f} mg/dL")
    logger.info(f"  - MAE: {metrics['MAE']:.2f} mg/dL")
    logger.info(f"  - MARD: {metrics['MARD']:.2f}%")
    
    # 2. SEG 분석 수행
    seg_results = evaluate_seg_for_model(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        results_dir=results_dir,
        experiment_name=experiment_name
    )
    
    # 3. 통합 결과 저장 (선택적)
    if save_detailed_results:
        # 상세 결과를 DataFrame으로 저장
        detailed_data = {
            'Reference_BG': y_true,
            'Predicted_BG': y_pred,
            'Absolute_Error': np.abs(y_true - y_pred),
            'Relative_Error_%': np.abs((y_pred - y_true) / y_true) * 100,
            'SEG_Zone': seg_results['zones']
        }
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = results_dir / f"detailed_results_{model_name.lower().replace(' ', '_')}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        logger.info(f"Detailed results saved: {detailed_path}")
    
    return {
        'metrics': metrics,
        'seg_results': seg_results
    }


def create_combined_summary(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Path
) -> pd.DataFrame:
    """
    모든 모델의 평가 결과를 하나의 요약 테이블로 통합
    
    Args:
        all_results: 모델별 평가 결과 딕셔너리
        save_path: 저장 경로
        
    Returns:
        pd.DataFrame: 통합 요약 테이블
    """
    summary_rows = []
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        seg_stats = results['seg_results']['statistics']
        
        row = {
            'Model': model_name,
            'R²': f"{metrics['R2']:.4f}",
            'RMSE (mg/dL)': f"{metrics['RMSE']:.2f}",
            'MAE (mg/dL)': f"{metrics['MAE']:.2f}",
            'MARD (%)': f"{metrics['MARD']:.2f}",
            'SEG None-Risk (%)': f"{seg_stats['None-Risk']:.2f}",
            'SEG Clinically-Acceptable (%)': f"{seg_stats['Clinically-Acceptable']:.2f}",
            'SEG Extreme-Risk (%)': f"{seg_stats['Total-Extreme-Risk']:.2f}"
        }
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    
    logger.info(f"Combined summary saved: {save_path}")
    logger.info("\n" + summary_df.to_string(index=False))
    
    return summary_df


def compare_seg_across_models(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Path,
    experiment_name: str = ""
) -> None:
    """
    모든 모델의 SEG 결과를 비교하는 시각화 생성
    
    Args:
        all_results: 모델별 평가 결과
        save_path: 저장 경로
        experiment_name: 실험 이름
    """
    # 모델별 SEG 통계 추출
    all_statistics = {}
    for model_name, results in all_results.items():
        all_statistics[model_name] = results['seg_results']['statistics']
    
    # SEG 비교 시각화 생성
    seg = SurveillanceErrorGrid()
    seg.plot_seg_comparison(
        all_statistics=all_statistics,
        save_path=save_path,
        title_suffix=experiment_name
    )
    
    logger.info(f"SEG comparison plot saved: {save_path}")