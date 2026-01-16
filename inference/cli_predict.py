# inference/cli_predict.py
# ============================================================
# CLI 기반 SG → BG 추론 스크립트
# - 사용자가 터미널에서 직접 값 입력
# - 학습 시 사용한 feature_builder 로직 그대로 재사용
# - SG_PLUS_META 기준 추론
# ============================================================

import pandas as pd # DataFrame 처리 라이브러리
import joblib # 학습된 모델 .pkl 형태로 저장/로드하기 위한 라이브러리
import os # 운영체제/경로 관련 유틸 사용

from src.feature_builder import build_features # 학습/추론에서 동일한 feature 생성규칙을 재사용

# ------------------------------------------------------------
# 추론용 모델 클래스
# ------------------------------------------------------------
class BGPredictor:
    """
    학습된 LightGBM 모델을 불러와
    사용자 입력 → BG 예측을 수행하는 클래스
    """

    def __init__(self, model_path: str):
        # 모델 경로 저장
        self.model_path = model_path

        # joblib으로 학습된 모델 로드
        self.model = joblib.load(model_path)

    def predict(self, input_df: pd.DataFrame) -> float:
        """
        입력 DataFrame을 feature_builder에 통과시켜
        BG 예측값을 반환
        """

        # build_features는 (X, y, feature_names)를 반환
        # 추론이므로 y는 None
        X, _, _ = build_features(
            input_df,
            mode="sg_plus_meta" # feature 구성 모드
        )

        # 모델 예측 (배열 형태 → 첫 값만 사용)
        bg_pred = self.model.predict(X)[0]

        return bg_pred


# ------------------------------------------------------------
# CLI 입력부
# ------------------------------------------------------------
def main():
    print("\n🧪 SG → BG CLI Prediction\n")

    # --------------------------------------------------------
    # 사용자 입력 받기
    # --------------------------------------------------------
    sg = float(input("SG (Salivary Glucose): "))

    meal_status = input(
        "Meal_Status (Fasting / Postprandial): "
    )

    bmi_class = input(
        "BMI_Class (Normal / Overweight / Obese / Healthy_Obesity / Skinny_Diabetes): "
    )

    age_group = input(
        "Age_Group (Young / Middle / Elderly): "
    )

    exercise = input(
        "Exercise (Sedentary / Moderate / High): "
    )

    family_history = input(
        "Family_History (None / Other / Diabetes): "
    )

    pregnancy = input(
        "Pregnancy (Not_Pregnant / Pregnant_Normal / Pregnant_GDM): "
    )

    # --------------------------------------------------------
    # 입력값을 DataFrame으로 구성
    # (컬럼명은 학습 데이터와 반드시 동일해야 함)
    # --------------------------------------------------------
    input_data = pd.DataFrame([{
        "SG": sg,
        "Meal_Status": meal_status,
        "BMI_Class": bmi_class,
        "Age_Group": age_group,
        "Exercise": exercise,
        "Family_History": family_history,
        "Pregnancy": pregnancy
    }])

    # --------------------------------------------------------
    # 모델 로드 및 예측
    # --------------------------------------------------------
    model_path = os.path.join(
        "results",
        "SG_PLUS_META",
        "best_model_lightgbm.pkl"
    )

    predictor = BGPredictor(model_path)

    bg_pred = predictor.predict(input_data)

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------
    print("\n✅ Prediction Result")
    print(f"➡️  Predicted BG: {bg_pred:.2f} mg/dL\n")


# ------------------------------------------------------------
# 스크립트 직접 실행 시 main() 호출
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
