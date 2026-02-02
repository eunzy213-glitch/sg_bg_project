"""
Surveillance Error Grid (SEG) 분석 모듈

임상적 평가 고도화를 위한 SEG 분석 수행
- SEG Zone 분류 (None-Risk, Slight-Risk, Moderate-Risk, Great-Risk, Extreme-Risk)
- 정확도 대비 위험도 기반 임상 평가
- 시각화 및 통계 요약
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# 로거 설정
logger = logging.getLogger(__name__)


class SurveillanceErrorGrid:
    """
    Surveillance Error Grid (SEG) 분석 클래스
    
    SEG는 Clarke Error Grid를 확장한 개념으로,
    연속 혈당 모니터링(CGM)의 정확도를 임상적 위험도 관점에서 평가합니다.
    """
    
    def __init__(self):
        """
        SEG 분석 초기화
        
        SEG Zone 정의:
        - None-Risk: 임상적 위험 없음
        - Slight-Risk, Lower/Upper: 낮은 위험
        - Moderate-Risk, Lower/Upper: 중간 위험  
        - Great-Risk, Lower/Upper: 높은 위험
        - Extreme-Risk, Lower/Upper: 매우 높은 위험
        """
        self.zone_colors = {
            'None-Risk': '#00FF00',           # 녹색 - 위험 없음
            'Slight-Risk-Lower': '#ADFF2F',   # 연두색 - 낮은 위험(저혈당)
            'Slight-Risk-Upper': '#FFFF00',   # 노란색 - 낮은 위험(고혈당)
            'Moderate-Risk-Lower': '#FFA500', # 주황색 - 중간 위험(저혈당)
            'Moderate-Risk-Upper': '#FF8C00', # 진한 주황 - 중간 위험(고혈당)
            'Great-Risk-Lower': '#FF4500',    # 빨강-주황 - 높은 위험(저혈당)
            'Great-Risk-Upper': '#FF0000',    # 빨강 - 높은 위험(고혈당)
            'Extreme-Risk-Lower': '#8B0000',  # 암적색 - 극심한 위험(저혈당)
            'Extreme-Risk-Upper': '#DC143C'   # 진홍색 - 극심한 위험(고혈당)
        }
        
    def classify_seg_zone(
        self, 
        reference: float, 
        prediction: float
    ) -> str:
        """
        SEG Zone 분류 수행
        
        Args:
            reference: 실제 BG 값 (mg/dL)
            prediction: 예측 BG 값 (mg/dL)
            
        Returns:
            str: SEG Zone 이름
            
        Notes:
            - 저혈당(<70), 정상(70-180), 고혈당(>180) 구간별로 다른 기준 적용
            - 상대 오차(% difference)와 절대 오차(mg/dL difference) 모두 고려
        """
        # 절대 오차 및 상대 오차 계산
        abs_diff = abs(prediction - reference)
        rel_diff = abs_diff / reference * 100 if reference > 0 else 0
        
        # 저혈당 구간 (<70 mg/dL)
        if reference < 70:
            if abs_diff <= 5:
                return 'None-Risk'
            elif abs_diff <= 15:
                return 'Slight-Risk-Lower'
            elif abs_diff <= 25:
                return 'Moderate-Risk-Lower'
            elif abs_diff <= 35:
                return 'Great-Risk-Lower'
            else:
                return 'Extreme-Risk-Lower'
        
        # 정상 혈당 구간 (70-180 mg/dL)
        elif 70 <= reference <= 180:
            if rel_diff <= 5:
                return 'None-Risk'
            elif rel_diff <= 10:
                if prediction < reference:
                    return 'Slight-Risk-Lower'
                else:
                    return 'Slight-Risk-Upper'
            elif rel_diff <= 20:
                if prediction < reference:
                    return 'Moderate-Risk-Lower'
                else:
                    return 'Moderate-Risk-Upper'
            elif rel_diff <= 30:
                if prediction < reference:
                    return 'Great-Risk-Lower'
                else:
                    return 'Great-Risk-Upper'
            else:
                if prediction < reference:
                    return 'Extreme-Risk-Lower'
                else:
                    return 'Extreme-Risk-Upper'
        
        # 고혈당 구간 (>180 mg/dL)
        else:
            if abs_diff <= 10:
                return 'None-Risk'
            elif abs_diff <= 25:
                if prediction < reference:
                    return 'Slight-Risk-Lower'
                else:
                    return 'Slight-Risk-Upper'
            elif abs_diff <= 45:
                if prediction < reference:
                    return 'Moderate-Risk-Lower'
                else:
                    return 'Moderate-Risk-Upper'
            elif abs_diff <= 65:
                if prediction < reference:
                    return 'Great-Risk-Lower'
                else:
                    return 'Great-Risk-Upper'
            else:
                if prediction < reference:
                    return 'Extreme-Risk-Lower'
                else:
                    return 'Extreme-Risk-Upper'
    
    def analyze_seg(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        전체 데이터셋에 대한 SEG 분석 수행
        
        Args:
            y_true: 실제 BG 값 배열
            y_pred: 예측 BG 값 배열
            
        Returns:
            zones: 각 샘플의 SEG Zone 배열
            statistics: Zone별 비율 및 통계
        """
        # Zone 분류 수행
        zones = np.array([
            self.classify_seg_zone(ref, pred) 
            for ref, pred in zip(y_true, y_pred)
        ])
        
        # Zone별 통계 계산
        total_samples = len(zones)
        statistics = {}
        
        # 각 Zone별 비율 계산
        for zone_name in self.zone_colors.keys():
            count = np.sum(zones == zone_name)
            percentage = (count / total_samples) * 100
            statistics[zone_name] = percentage
        
        # 위험도별 그룹 통계 계산
        statistics['Total-None-Risk'] = statistics['None-Risk']
        
        statistics['Total-Slight-Risk'] = (
            statistics['Slight-Risk-Lower'] + 
            statistics['Slight-Risk-Upper']
        )
        
        statistics['Total-Moderate-Risk'] = (
            statistics['Moderate-Risk-Lower'] + 
            statistics['Moderate-Risk-Upper']
        )
        
        statistics['Total-Great-Risk'] = (
            statistics['Great-Risk-Lower'] + 
            statistics['Great-Risk-Upper']
        )
        
        statistics['Total-Extreme-Risk'] = (
            statistics['Extreme-Risk-Lower'] + 
            statistics['Extreme-Risk-Upper']
        )
        
        # 임상적 허용 범위 (None + Slight Risk)
        statistics['Clinically-Acceptable'] = (
            statistics['Total-None-Risk'] + 
            statistics['Total-Slight-Risk']
        )
        
        logger.info(f"SEG Analysis completed: {total_samples} samples analyzed")
        logger.info(f"Clinically Acceptable: {statistics['Clinically-Acceptable']:.2f}%")
        
        return zones, statistics
    
    def plot_seg(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        zones: np.ndarray,
        model_name: str,
        save_path: Path,
        title_suffix: str = ""
    ) -> None:
        """
        SEG 시각화 생성 및 저장
        
        Args:
            y_true: 실제 BG 값
            y_pred: 예측 BG 값  
            zones: SEG Zone 분류 결과
            model_name: 모델 이름
            save_path: 저장 경로
            title_suffix: 제목 접미사 (예: "SG_ONLY")
        """
        # Figure 설정
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Zone별로 점 그리기
        for zone_name, color in self.zone_colors.items():
            mask = zones == zone_name
            if np.any(mask):
                ax.scatter(
                    y_true[mask], 
                    y_pred[mask],
                    c=color,
                    label=zone_name,
                    alpha=0.6,
                    s=30,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        # 대각선 (Perfect Prediction Line)
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        ax.plot(
            [min_val, max_val], 
            [min_val, max_val], 
            'k--', 
            linewidth=2, 
            label='Perfect Prediction',
            alpha=0.5
        )
        
        # Zone 경계선 표시 (주요 임계값)
        # 저혈당 경계 (70 mg/dL)
        ax.axvline(x=70, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=70, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # 고혈당 경계 (180 mg/dL)
        ax.axvline(x=180, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=180, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # 축 레이블 및 제목
        ax.set_xlabel('Reference BG (mg/dL)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted BG (mg/dL)', fontsize=14, fontweight='bold')
        
        title = f'Surveillance Error Grid - {model_name}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 범례 설정 (두 열로 배치)
        ax.legend(
            loc='upper left',
            fontsize=9,
            ncol=2,
            framealpha=0.9,
            edgecolor='black'
        )
        
        # 그리드 추가
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SEG plot saved: {save_path}")
    
    def create_seg_summary_table(
        self,
        statistics: Dict[str, float],
        model_name: str,
        save_path: Path
    ) -> pd.DataFrame:
        """
        SEG 통계 요약 테이블 생성
        
        Args:
            statistics: SEG 통계 딕셔너리
            model_name: 모델 이름
            save_path: CSV 저장 경로
            
        Returns:
            pd.DataFrame: 요약 테이블
        """
        # 요약 데이터 생성
        summary_data = {
            'Model': [model_name],
            'None-Risk (%)': [f"{statistics['None-Risk']:.2f}"],
            'Slight-Risk (%)': [f"{statistics['Total-Slight-Risk']:.2f}"],
            'Moderate-Risk (%)': [f"{statistics['Total-Moderate-Risk']:.2f}"],
            'Great-Risk (%)': [f"{statistics['Total-Great-Risk']:.2f}"],
            'Extreme-Risk (%)': [f"{statistics['Total-Extreme-Risk']:.2f}"],
            'Clinically-Acceptable (%)': [f"{statistics['Clinically-Acceptable']:.2f}"]
        }
        
        df = pd.DataFrame(summary_data)
        
        # CSV 저장
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        
        logger.info(f"SEG summary table saved: {save_path}")
        
        return df
    
    def plot_seg_comparison(
        self,
        all_statistics: Dict[str, Dict[str, float]],
        save_path: Path,
        title_suffix: str = ""
    ) -> None:
        """
        여러 모델의 SEG 통계 비교 시각화
        
        Args:
            all_statistics: 모델별 SEG 통계 딕셔너리
            save_path: 저장 경로
            title_suffix: 제목 접미사
        """
        # 데이터 준비
        models = list(all_statistics.keys())
        risk_categories = [
            'None-Risk',
            'Total-Slight-Risk',
            'Total-Moderate-Risk',
            'Total-Great-Risk',
            'Total-Extreme-Risk'
        ]
        
        # 각 위험도별 데이터 추출
        data = {cat: [] for cat in risk_categories}
        for model in models:
            stats = all_statistics[model]
            for cat in risk_categories:
                data[cat].append(stats[cat])
        
        # 누적 막대 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.6
        
        # 색상 정의
        colors = {
            'None-Risk': '#00FF00',
            'Total-Slight-Risk': '#FFFF00',
            'Total-Moderate-Risk': '#FFA500',
            'Total-Great-Risk': '#FF0000',
            'Total-Extreme-Risk': '#8B0000'
        }
        
        # 누적 막대 그리기
        bottom = np.zeros(len(models))
        for cat in risk_categories:
            ax.bar(
                x, 
                data[cat], 
                width,
                bottom=bottom,
                label=cat,
                color=colors[cat],
                edgecolor='black',
                linewidth=1
            )
            bottom += np.array(data[cat])
        
        # 축 레이블 및 제목
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        
        title = 'SEG Risk Distribution Comparison'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 100)
        
        # 그리드 추가
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SEG comparison plot saved: {save_path}")


def evaluate_seg_for_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    results_dir: Path,
    experiment_name: str = ""
) -> Dict[str, Any]:
    """
    단일 모델에 대한 SEG 평가 수행
    
    Args:
        y_true: 실제 BG 값
        y_pred: 예측 BG 값
        model_name: 모델 이름
        results_dir: 결과 저장 디렉토리
        experiment_name: 실험 이름 (예: "SG_ONLY", "SG_PLUS_META")
        
    Returns:
        Dict: SEG 분석 결과 (zones, statistics)
    """
    logger.info(f"Starting SEG evaluation for {model_name}...")
    
    # SEG 분석 객체 생성
    seg = SurveillanceErrorGrid()
    
    # SEG 분석 수행
    zones, statistics = seg.analyze_seg(y_true, y_pred)
    
    # SEG 시각화 저장
    seg_plot_path = results_dir / f"seg_{model_name.lower().replace(' ', '_')}.png"
    seg.plot_seg(
        y_true=y_true,
        y_pred=y_pred,
        zones=zones,
        model_name=model_name,
        save_path=seg_plot_path,
        title_suffix=experiment_name
    )
    
    # 통계 테이블 저장
    table_path = results_dir / f"seg_{model_name.lower().replace(' ', '_')}_summary.csv"
    summary_df = seg.create_seg_summary_table(
        statistics=statistics,
        model_name=model_name,
        save_path=table_path
    )
    
    logger.info(f"SEG evaluation completed for {model_name}")
    logger.info(f"  - Clinically Acceptable: {statistics['Clinically-Acceptable']:.2f}%")
    logger.info(f"  - None-Risk: {statistics['None-Risk']:.2f}%")
    logger.info(f"  - Extreme-Risk: {statistics['Total-Extreme-Risk']:.2f}%")
    
    return {
        'zones': zones,
        'statistics': statistics,
        'summary_df': summary_df
    }