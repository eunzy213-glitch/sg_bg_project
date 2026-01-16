# inference/inference_model.py
# ============================================================
# BG 추론 전용 모델 클래스
# ============================================================

import pandas as pd # DataFrame 처리 라이브러리
import joblib # 학습된 모델 .pkl 형태로 저장/로드하기 위한 라이브러리

from src.feature_builder import build_features # 학습/추론에서 동일한 feature 생성규칙을 재사용


class BGPredictor: # SG + Meta 정보를 입력받아 BG를 예측하는 클래스

    def __init__(self, model_path: str):
 
        self.model = joblib.load(model_path) # .pkl 모델 파일을 메모리에 로드해서 self.model에 저장

    def predict(self, input_dict: dict) -> float: # BG 예측 수행
        """
        BG 예측 수행

        Parameters
        ----------
        input_dict : dict
            {
              "SG": 108,
              "Meal_Status": "Fasting",
              "BMI_Class": "Overweight",
              "Age_Group": "Middle",
              "Exercise": "Moderate",
              "Family_History": "Diabetes",
              "Pregnancy": "Not_Applicable"
            }

        Returns
        -------
        float
            예측된 BG 값
        """

        # -----------------------------
        # 1️⃣ dict → DataFrame
        # -----------------------------
        df = pd.DataFrame([input_dict])

        # -----------------------------
        # 2️⃣ Feature 구성 (공통 로직)
        # -----------------------------
        X, _, _ = build_features(
            df,
            mode="SG_PLUS_META"
        )

        # -----------------------------
        # 3️⃣ 예측
        # -----------------------------
        bg_pred = self.model.predict(X)[0]

        return float(bg_pred)
