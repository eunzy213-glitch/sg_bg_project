# src/explainability.py
# ============================================================
# SHAP / LIME Explainability Utilities
# ============================================================

import os # 결과 이미지 저장을 위한 폴더/경로 처리
import numpy as np # 배열처리/슬라이싱 등 수치연산용
import shap # SHAP 라이브러리
import matplotlib.pyplot as plt # 시각화 라이브러리

from lime.lime_tabular import LimeTabularExplainer # LIME 라이브러리


def run_shap_analysis( # shap 분석 함수
    model,
    X_train,
    X_test,
    feature_names,
    save_dir,
    max_display=20
):

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # TreeExplainer (RF, LightGBM)
    # --------------------------------------------------------
    explainer = shap.TreeExplainer(model) # TreeExplainer는 트리 기반 모델에 대해 SHAP 값을 효율적으로 계산해주는 explainer

    X_sample = X_test[:200] # 계산량 제한

    shap_values = explainer.shap_values(X_sample)  # SHAP 값 계산

    # --------------------------------------------------------
    # SHAP Summary Plot
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        color_bar=False  
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "shap_summary.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def run_lime_analysis( # lime 분석 함수
    model,
    X_train,
    X_test,
    feature_names,
    save_dir,
    sample_idx=0, # 몇번째 샘플을 설명할지
    num_features=10 # 결과에서 상위 몇개 feature를 보여줄지
):

    os.makedirs(save_dir, exist_ok=True)

    explainer = LimeTabularExplainer( # lime explainer 객체 생성
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,
        random_state=42,
        sample_around_instance=True
    )

    exp = explainer.explain_instance( # 특정 샘플 1건 설명 생성
        X_test[sample_idx],
        model.predict,
        num_features=num_features,
        num_samples=5000
    )

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    fig.tight_layout()

    fig.savefig(
        os.path.join(save_dir, f"lime_sample_{sample_idx}.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)
