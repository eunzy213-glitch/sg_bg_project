"""
SEG 분석 모듈 테스트 스크립트

설치 및 동작을 확인하기 위한 간단한 테스트
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

from src.seg_analysis import SurveillanceErrorGrid, evaluate_seg_for_model


def test_seg_classification():
    """SEG Zone 분류 기능 테스트"""
    print("="*80)
    print("TEST 1: SEG Zone Classification")
    print("="*80)
    
    seg = SurveillanceErrorGrid()
    
    # 테스트 케이스들
    test_cases = [
        # (reference, prediction, expected_zone)
        (100, 100, "None-Risk"),            # 완벽한 예측, diff=0
        (100, 110, "Slight-Risk-Upper"),    # 정상 구간, rel_diff=10% -> Slight-Upper
        (100, 90, "Slight-Risk-Lower"),     # 정상 구간, rel_diff=10% -> Slight-Lower
        (60, 80, "Moderate-Risk-Lower"),    # 저혈당(<70), abs_diff=20 -> Moderate (15<20<25)
        (200, 250, "Great-Risk-Upper"),     # 고혈당(>180), abs_diff=50 -> Great (45<50<65)
        (70, 30, "Extreme-Risk-Lower"),     # 저혈당(<70), abs_diff=40 -> Extreme (>35)
        (180, 260, "Extreme-Risk-Upper"),   # 고혈당(>180), abs_diff=80 -> Extreme (>65)
    ]
    
    print("\nTest Cases:")
    print(f"{'Reference':>12} {'Predicted':>12} {'Expected Zone':>25} {'Actual Zone':>25} {'Status':>10}")
    print("-" * 100)
    
    passed = 0
    failed = 0
    
    for ref, pred, expected in test_cases:
        actual = seg.classify_seg_zone(ref, pred)
        status = "✓ PASS" if actual == expected else "✗ FAIL"
        
        if status == "✓ PASS":
            passed += 1
        else:
            failed += 1
        
        print(f"{ref:>12.1f} {pred:>12.1f} {expected:>25} {actual:>25} {status:>10}")
    
    print("\n" + "="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


def test_seg_analysis():
    """전체 SEG 분석 기능 테스트"""
    print("="*80)
    print("TEST 2: SEG Analysis with Synthetic Data")
    print("="*80)
    
    # 합성 데이터 생성
    np.random.seed(42)
    n_samples = 500
    
    # 각 구간별 샘플 수 정확히 계산
    n_low = n_samples // 3  # 166
    n_normal = n_samples // 3  # 166
    n_high = n_samples - n_low - n_normal  # 168 (나머지)
    
    # 다양한 혈당 범위의 데이터 생성
    y_true = np.concatenate([
        np.random.uniform(50, 70, n_low),     # 저혈당
        np.random.uniform(70, 180, n_normal), # 정상
        np.random.uniform(180, 300, n_high)   # 고혈당
    ])
    
    # 예측값 생성 (실제값 + 노이즈)
    noise = np.random.normal(0, 15, n_samples)
    y_pred = y_true + noise
    y_pred = np.clip(y_pred, 40, 350)  # 합리적인 범위로 제한
    
    print(f"\nSynthetic Data Generated:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Reference BG range: {y_true.min():.1f} - {y_true.max():.1f} mg/dL")
    print(f"  - Predicted BG range: {y_pred.min():.1f} - {y_pred.max():.1f} mg/dL")
    
    # SEG 분석 수행
    seg = SurveillanceErrorGrid()
    zones, statistics = seg.analyze_seg(y_true, y_pred)
    
    print("\nSEG Statistics:")
    print("-" * 80)
    print(f"{'Zone':<30} {'Percentage':>15}")
    print("-" * 80)
    
    # 주요 통계 출력
    key_stats = [
        'None-Risk',
        'Total-Slight-Risk',
        'Total-Moderate-Risk',
        'Total-Great-Risk',
        'Total-Extreme-Risk',
        'Clinically-Acceptable'
    ]
    
    for stat in key_stats:
        if stat in statistics:
            print(f"{stat:<30} {statistics[stat]:>14.2f}%")
    
    print("-" * 80)
    
    # 임상적 허용 기준 검사
    acceptable = statistics['Clinically-Acceptable']
    extreme = statistics['Total-Extreme-Risk']
    
    print(f"\nClinical Evaluation:")
    if acceptable >= 85:
        print(f"  ✓ Clinically-Acceptable: {acceptable:.2f}% (≥85% target) - GOOD")
    else:
        print(f"  ⚠ Clinically-Acceptable: {acceptable:.2f}% (<85% target) - NEEDS IMPROVEMENT")
    
    if extreme <= 1:
        print(f"  ✓ Extreme-Risk: {extreme:.2f}% (≤1% target) - SAFE")
    else:
        print(f"  ⚠ Extreme-Risk: {extreme:.2f}% (>1% target) - CAUTION")
    
    print("\n" + "="*80 + "\n")
    
    return zones, statistics


def test_seg_visualization():
    """SEG 시각화 테스트"""
    print("="*80)
    print("TEST 3: SEG Visualization")
    print("="*80)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 300
    
    y_true = np.random.uniform(60, 250, n_samples)
    y_pred = y_true + np.random.normal(0, 20, n_samples)
    y_pred = np.clip(y_pred, 50, 300)
    
    # 결과 디렉토리 생성
    test_results_dir = Path("test_results")
    test_results_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating SEG visualizations...")
    print(f"Output directory: {test_results_dir}")
    
    # SEG 분석 및 시각화
    seg = SurveillanceErrorGrid()
    zones, statistics = seg.analyze_seg(y_true, y_pred)
    
    # 시각화 저장
    plot_path = test_results_dir / "test_seg_plot.png"
    seg.plot_seg(
        y_true=y_true,
        y_pred=y_pred,
        zones=zones,
        model_name="Test Model",
        save_path=plot_path,
        title_suffix="TEST"
    )
    
    # 요약 테이블 저장
    table_path = test_results_dir / "test_seg_summary.csv"
    summary_df = seg.create_seg_summary_table(
        statistics=statistics,
        model_name="Test Model",
        save_path=table_path
    )
    
    print("\nGenerated Files:")
    print(f"  ✓ SEG Plot: {plot_path}")
    print(f"  ✓ Summary Table: {table_path}")
    
    # 파일 존재 확인
    if plot_path.exists() and table_path.exists():
        print("\n✓ All files created successfully!")
        return True
    else:
        print("\n✗ File creation failed!")
        return False


def test_model_evaluation():
    """전체 모델 평가 프로세스 테스트"""
    print("="*80)
    print("TEST 4: Complete Model Evaluation with SEG")
    print("="*80)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 400
    
    # 각 구간별 샘플 수 정확히 계산
    n_low = n_samples // 4  # 100
    n_normal = n_samples // 2  # 200
    n_high = n_samples - n_low - n_normal  # 100
    
    y_true = np.concatenate([
        np.random.uniform(50, 70, n_low),
        np.random.uniform(70, 180, n_normal),
        np.random.uniform(180, 280, n_high)
    ])
    
    # 두 가지 모델 시뮬레이션
    models = {
        'Good Model': y_true + np.random.normal(0, 10, n_samples),
        'Poor Model': y_true + np.random.normal(0, 30, n_samples)
    }
    
    # 결과 디렉토리
    test_results_dir = Path("test_results")
    
    print(f"\nEvaluating {len(models)} models...")
    
    all_results = {}
    
    for model_name, y_pred in models.items():
        y_pred = np.clip(y_pred, 40, 320)
        
        print(f"\n--- {model_name} ---")
        
        result = evaluate_seg_for_model(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            results_dir=test_results_dir,
            experiment_name="TEST"
        )
        
        all_results[model_name] = result
        
        # 주요 지표 출력
        stats = result['statistics']
        print(f"  Clinically-Acceptable: {stats['Clinically-Acceptable']:.2f}%")
        print(f"  Extreme-Risk: {stats['Total-Extreme-Risk']:.2f}%")
    
    # 비교 시각화
    print("\nGenerating comparison plot...")
    
    all_statistics = {name: result['statistics'] for name, result in all_results.items()}
    
    comparison_path = test_results_dir / "test_seg_comparison.png"
    seg = SurveillanceErrorGrid()
    seg.plot_seg_comparison(
        all_statistics=all_statistics,
        save_path=comparison_path,
        title_suffix="TEST"
    )
    
    print(f"✓ Comparison plot saved: {comparison_path}")
    
    print("\n" + "="*80 + "\n")
    
    return all_results


def main():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print(" "*20 + "SEG ANALYSIS MODULE TEST SUITE")
    print("="*80 + "\n")
    
    test_results = []
    
    # 테스트 1: Zone 분류
    try:
        result = test_seg_classification()
        test_results.append(("Zone Classification", result))
    except Exception as e:
        print(f"✗ Test 1 failed with error: {e}")
        test_results.append(("Zone Classification", False))
    
    # 테스트 2: SEG 분석
    try:
        zones, stats = test_seg_analysis()
        test_results.append(("SEG Analysis", True))
    except Exception as e:
        print(f"✗ Test 2 failed with error: {e}")
        test_results.append(("SEG Analysis", False))
    
    # 테스트 3: 시각화
    try:
        result = test_seg_visualization()
        test_results.append(("Visualization", result))
    except Exception as e:
        print(f"✗ Test 3 failed with error: {e}")
        test_results.append(("Visualization", False))
    
    # 테스트 4: 모델 평가
    try:
        results = test_model_evaluation()
        test_results.append(("Model Evaluation", True))
    except Exception as e:
        print(f"✗ Test 4 failed with error: {e}")
        test_results.append(("Model Evaluation", False))
    
    # 최종 결과 요약
    print("\n" + "="*80)
    print(" "*25 + "TEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Test Name':<30} {'Result':>20}")
    print("-" * 80)
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_name, passed in test_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status:>20}")
        if passed:
            passed_count += 1
    
    print("-" * 80)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("="*80 + "\n")
    
    if passed_count == total_count:
        print("🎉 All tests passed! SEG analysis module is ready to use.")
        print("\nNext steps:")
        print("1. Review the generated files in 'test_results/' directory")
        print("2. Follow SEG_INTEGRATION_GUIDE.md to integrate into your pipeline")
        print("3. Run your actual experiments with SEG analysis enabled")
    else:
        print("⚠ Some tests failed. Please review the error messages above.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)