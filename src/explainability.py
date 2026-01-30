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

# ============================================================
# 🆕 SHAP Interaction Value 분석 함수
# - 트리 기반 모델(XGBoost / LightGBM / CatBoost)에서
#   feature 간 상호작용 기여도를 계산하고 저장합니다.
# ============================================================

import os  # 파일 경로/폴더 생성용
import numpy as np  # 수치 계산용
import matplotlib.pyplot as plt  # 정적 시각화용
import shap  # SHAP 분석 라이브러리


def run_shap_interaction_analysis(
    model,                 # 학습된 모델 객체 (Tree 기반 권장)
    X_train,               # 학습 데이터 (numpy array)
    feature_names,         # 컬럼명 리스트 (X_train의 열 순서와 동일해야 함)
    save_dir,              # 결과 저장 폴더
    sample_size=500,       # interaction 계산 시 사용할 샘플 수 (크면 느리고 메모리 큼)
    top_k=20,              # 상호작용 상위 몇 개를 요약할지
    random_state=42        # 샘플링 재현성 고정
):
    """
    SHAP Interaction Value 분석:
    - shap.TreeExplainer(model).shap_interaction_values(X) 로 interaction tensor를 얻습니다.
    - (n_samples, n_features, n_features) 형태이며,
      i,j 성분은 "feature i와 j의 상호작용 기여"를 의미합니다.
    - 대각선(i==j)은 단독(main effect) 성분이 들어갑니다.
    """

    # --------------------------------------------------------
    # 0️⃣ 저장 폴더 생성 (없으면 생성)
    # --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)  # save_dir이 없으면 생성, 있으면 통과

    # --------------------------------------------------------
    # 1️⃣ 입력 데이터 샘플링 (interaction은 O(F^2)라 매우 무거움)
    # --------------------------------------------------------
    rng = np.random.RandomState(random_state)  # 재현 가능한 랜덤 시드 생성

    n = X_train.shape[0]  # 전체 샘플 개수
    if n > sample_size:  # 샘플이 많으면 일부만 사용
        idx = rng.choice(n, size=sample_size, replace=False)  # 중복 없이 sample_size개 선택
        X_used = X_train[idx]  # 선택된 샘플만 사용
    else:
        X_used = X_train  # 샘플이 적으면 전체 사용

    # --------------------------------------------------------
    # 2️⃣ TreeExplainer 생성
    # --------------------------------------------------------
    # TreeExplainer는 트리 계열에 가장 잘 맞습니다.
    # - XGBoost / LightGBM / CatBoost / sklearn RF 등에서 주로 사용 가능
    # - 모델이 트리 기반이 아니면 여기서 실패할 수 있음
    try:
        explainer = shap.TreeExplainer(model)  # 트리 기반 SHAP explainer 생성
    except Exception as e:
        # 트리 모델이 아니거나, SHAP이 모델을 지원하지 않을 때 예외 발생 가능
        print(f"❌ SHAP Interaction 분석 불가: TreeExplainer 생성 실패 ({type(e).__name__}: {e})")
        return  # 실패 시 조용히 종료(파이프라인 전체가 죽지 않도록)

    # --------------------------------------------------------
    # 3️⃣ Interaction Value 계산
    # --------------------------------------------------------
    # shap_interaction_values 결과:
    # - 회귀(regression): (n_samples, n_features, n_features)
    # - 이진/다중 분류: 클래스별 리스트 형태일 수 있음
    try:
        inter = explainer.shap_interaction_values(X_used)  # interaction tensor 계산
    except Exception as e:
        print(f"❌ SHAP Interaction 계산 실패 ({type(e).__name__}: {e})")
        return

    # --------------------------------------------------------
    # 4️⃣ 분류 모델일 경우 shape 정리
    # --------------------------------------------------------
    # 분류 모델에서는 inter가 list로 나오는 경우가 있습니다.
    # 현재 프로젝트는 회귀(BG 예측)이므로 보통 ndarray일 텐데,
    # 안전하게 처리합니다.
    if isinstance(inter, list):
        # 예: 이진분류면 [class0_tensor, class1_tensor] 이런 구조가 올 수 있음
        # 여기서는 "마지막 클래스"를 선택하거나 평균을 낼 수 있습니다.
        # 회귀 프로젝트라면 이 분기는 거의 안 타지만, 방어 코드로 둡니다.
        inter = inter[-1]  # 관행적으로 positive class(또는 마지막 class) 선택

    # inter shape: (N, F, F)
    # --------------------------------------------------------
    # 5️⃣ 상호작용 강도 행렬(interaction strength matrix) 만들기
    # --------------------------------------------------------
    # 우리가 보고 싶은 건 "쌍(i,j) 상호작용이 전체적으로 얼마나 큰가"
    # -> 샘플 축 평균 + 절댓값 평균으로 요약하는 것이 일반적입니다.
    #
    # (i,j) 상호작용 강도 = mean(|inter[:, i, j]|)
    inter_abs_mean = np.mean(np.abs(inter), axis=0)  # shape: (F, F)

    # 대각선(inter_abs_mean[i,i])은 단독효과(main effect)를 의미하므로,
    # "순수 상호작용"만 보고 싶다면 대각선을 0으로 제거하는 것이 직관적입니다.
    inter_abs_mean_no_diag = inter_abs_mean.copy()  # 원본 보존을 위해 복사
    np.fill_diagonal(inter_abs_mean_no_diag, 0.0)  # 대각선만 0으로

    # --------------------------------------------------------
    # 6️⃣ Heatmap 저장 (전체 상호작용 구조를 한 눈에 보기)
    # --------------------------------------------------------
    plt.figure(figsize=(12, 10))  # 그림 크기 설정
    plt.imshow(inter_abs_mean_no_diag, aspect="auto")  # (F,F) 행렬을 이미지로 표현
    plt.colorbar(label="mean(|SHAP interaction|)")  # 컬러바 추가
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)  # x축 라벨
    plt.yticks(range(len(feature_names)), feature_names)  # y축 라벨
    plt.title("SHAP Interaction Strength Heatmap (mean absolute)")  # 제목
    plt.tight_layout()  # 레이아웃 자동 조정
    plt.savefig(os.path.join(save_dir, "shap_interaction_heatmap.png"), dpi=200)  # 파일 저장
    plt.close()  # 메모리 누수 방지

    # --------------------------------------------------------
    # 7️⃣ Top-K 상호작용 쌍 추출
    # --------------------------------------------------------
    # (i,j)와 (j,i)는 대칭이므로, i<j 상삼각만 사용합니다.
    F = len(feature_names)  # feature 개수
    pairs = []  # (score, i, j) 저장할 리스트

    for i in range(F):  # 첫 번째 feature index
        for j in range(i + 1, F):  # 두 번째 feature index (i보다 큰 것만)
            score = inter_abs_mean_no_diag[i, j]  # 해당 상호작용 강도
            pairs.append((score, i, j))  # 리스트에 추가

    # score 기준으로 내림차순 정렬(큰 상호작용이 상위로)
    pairs.sort(key=lambda x: x[0], reverse=True)

    # 상위 top_k개만 선택
    top_pairs = pairs[:top_k]

    # --------------------------------------------------------
    # 8️⃣ Top-K 상호작용 Bar Plot 저장
    # --------------------------------------------------------
    labels = []  # 막대 라벨(예: "SG × BMI_Class_Obese")
    values = []  # 막대 값(상호작용 강도)

    for score, i, j in top_pairs:  # top pair 순회
        labels.append(f"{feature_names[i]} × {feature_names[j]}")  # 쌍 라벨 생성
        values.append(score)  # 값 추가

    plt.figure(figsize=(12, 6))  # 그림 크기 설정
    y_pos = np.arange(len(values))  # y축 인덱스
    plt.barh(y_pos, values)  # 가로 막대 그래프
    plt.yticks(y_pos, labels)  # y축 라벨 설정
    plt.gca().invert_yaxis()  # 가장 큰 값이 위에 오도록 뒤집기
    plt.xlabel("mean(|SHAP interaction|)")  # x축 라벨
    plt.title(f"Top-{top_k} SHAP Interaction Pairs")  # 제목
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(os.path.join(save_dir, "shap_interaction_topk.png"), dpi=200)  # 파일 저장
    plt.close()  # 닫기

    # --------------------------------------------------------
    # 9️⃣ Top-K 상호작용을 CSV로도 저장 (README에 옮기기 쉬움)
    # --------------------------------------------------------
    csv_path = os.path.join(save_dir, "shap_interaction_topk.csv")  # 저장 경로
    with open(csv_path, "w", encoding="utf-8") as f:  # 파일 열기
        f.write("rank,feature_i,feature_j,interaction_strength\n")  # 헤더 작성
        for rank, (score, i, j) in enumerate(top_pairs, start=1):  # 1부터 순위 매김
            f.write(f"{rank},{feature_names[i]},{feature_names[j]},{score}\n")  # 한 줄씩 기록

    print(f"✅ SHAP Interaction 분석 저장 완료: {save_dir}")  # 완료 로그 출력
    
def save_shap_interaction_heatmap(
    model,
    X,
    feature_names,
    save_dir,
    max_display=15
):
    """
    SHAP Interaction Value 기반 Heatmap 저장 함수

    Parameters
    ----------
    model : 학습된 트리 기반 모델 (LightGBM / XGBoost / CatBoost)
    X : ndarray
        모델 입력 feature (Explain용 One-Hot Feature)
    feature_names : list
        feature 이름 리스트
    save_dir : str
        결과 이미지 저장 폴더
    max_display : int
        시각화에 사용할 상위 feature 개수
    """

    # --------------------------------------------------------
    # 1️⃣ SHAP Explainer 생성 (Tree 기반 모델 전용)
    # --------------------------------------------------------
    explainer = shap.TreeExplainer(model)

    # --------------------------------------------------------
    # 2️⃣ SHAP Interaction Value 계산
    # 결과 shape: (n_samples, n_features, n_features)
    # --------------------------------------------------------
    interaction_values = explainer.shap_interaction_values(X)

    # --------------------------------------------------------
    # 3️⃣ 샘플 평균 → 전역 Interaction Matrix 생성
    # shape: (n_features, n_features)
    # --------------------------------------------------------
    interaction_mean = np.mean(np.abs(interaction_values), axis=0)

    # --------------------------------------------------------
    # 4️⃣ 중요도 기준 상위 feature 선택
    # (대각선 = main effect)
    # --------------------------------------------------------
    main_effect = np.diag(interaction_mean)
    top_idx = np.argsort(main_effect)[::-1][:max_display]

    interaction_top = interaction_mean[np.ix_(top_idx, top_idx)]
    feature_top = [feature_names[i] for i in top_idx]

    # --------------------------------------------------------
    # 5️⃣ Heatmap 시각화
    # --------------------------------------------------------
    plt.figure(figsize=(10, 8))
    im = plt.imshow(interaction_top, cmap="Reds")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(
        range(len(feature_top)),
        feature_top,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(feature_top)),
        feature_top
    )

    plt.title("SHAP Interaction Value Heatmap")

    plt.tight_layout()

    # --------------------------------------------------------
    # 6️⃣ 저장
    # --------------------------------------------------------
    save_path = os.path.join(save_dir, "shap_interaction_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ SHAP Interaction Heatmap 저장 완료: {save_path}")    
