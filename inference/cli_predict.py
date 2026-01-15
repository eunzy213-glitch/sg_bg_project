# inference/cli_predict.py
# ============================================================
# CLI 기반 실시간 BG 예측
# ============================================================

from inference.inference_model import BGPredictor


def main():

    print("\n=== SG → BG 실시간 예측 ===\n")

    # -----------------------------
    # 1️⃣ 사용자 입력
    # -----------------------------
    input_data = {
        "SG": float(input("SG 입력: ")),

        "Meal_Status": input(
            "Meal_Status (Fasting / Postprandial): "
        ).strip(),

        "BMI_Class": input(
            "BMI_Class (Normal / Obese / Overweight / Healthy_Obesity / Skinny_Diabetes): "
        ).strip(),

        "Age_Group": input(
            "Age_Group (Young / Middle / Elderly): "
        ).strip(),

        "Exercise": input(
            "Exercise (Sedentary / Moderate / High): "
        ).strip(),

        "Family_History": input(
            "Family_History (None / Other / Diabetes): "
        ).strip(),

        "Pregnancy": input(
            "Pregnancy (Not_Applicable / None / Pregnant_Normal): "
        ).strip()
    }

    # -----------------------------
    # 2️⃣ 모델 로드 + 예측
    # -----------------------------
    predictor = BGPredictor(
        model_path="results/SG_PLUS_META/best_model_lightgbm.pkl"
    )

    bg = predictor.predict(input_data)

    print("\n👉 예측된 BG:", round(bg, 2))


if __name__ == "__main__":
    main()
