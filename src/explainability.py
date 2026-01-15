# src/explainability.py
# ============================================================
# SHAP / LIME Explainability Utilities
# ============================================================

import os
import numpy as np
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


def run_shap_analysis(
    model,
    X_train,
    X_test,
    feature_names,
    save_dir,
    max_display=20
):
    """
    SHAP Global Explanation (Tree-based models)

    Parameters
    ----------
    model : trained model
    X_train : np.ndarray
        학습에 사용된 feature (explainer 기준)
    X_test : np.ndarray
        SHAP 값을 계산할 feature
    feature_names : list
        feature 이름
    save_dir : str
        결과 저장 경로
    max_display : int
        SHAP summary에 표시할 최대 feature 수
    """

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # TreeExplainer (RF, LightGBM)
    # --------------------------------------------------------
    explainer = shap.TreeExplainer(model)

    # 계산량 제한
    X_sample = X_test[:200]

    shap_values = explainer.shap_values(X_sample)

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
        color_bar=False  # 서버/비GUI 환경 안정성
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "shap_summary.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def run_lime_analysis(
    model,
    X_train,
    X_test,
    feature_names,
    save_dir,
    sample_idx=0,
    num_features=10
):
    """
    LIME Local Explanation (single sample)

    Parameters
    ----------
    model : trained model
    X_train : np.ndarray
        학습 데이터
    X_test : np.ndarray
        설명할 데이터
    feature_names : list
        feature 이름
    save_dir : str
        결과 저장 경로
    sample_idx : int
        설명할 샘플 index
    num_features : int
        표시할 feature 개수
    """

    os.makedirs(save_dir, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False
    )

    exp = explainer.explain_instance(
        X_test[sample_idx],
        model.predict,
        num_features=num_features
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
