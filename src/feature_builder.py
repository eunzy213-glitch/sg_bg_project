import pandas as pd


ENCODING_MAP = {
    "Meal_Status": {
        "Fasting": 0,
        "Postprandial": 1
    },
    "BMI_Class": {
        "Normal": 0,
        "Overweight": 1,
        "Obese": 2,
        "Healthy_Obesity": 3,
        "Skinny_Diabetes": 4
    },
    "Age_Group": {
        "Young": 0,
        "Middle": 1,
        "Elderly": 2
    },
    "Exercise": {
        "Sedentary": 0,
        "Moderate": 1,
        "High": 2
    },
    "Family_History": {
        "None": 0,
        "Other": 1,
        "Diabetes": 2
    },
    "Pregnancy": {
        "Not_Applicable": 0,
        "None": 1,
        "Pregnant_Normal": 2
    }
}


def normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 범주형 컬럼을
    ENCODING_MAP의 key 중 하나로만 강제 정규화
    """

    df = df.copy()

    # Pregnancy만 특별 처리 (가장 문제 많음)
    if "Pregnancy" in df.columns:

        # 1️⃣ 진짜 NaN / None / 공백 전부 통일
        s = df["Pregnancy"].fillna("Not_Applicable").astype(str).str.strip().str.lower()

        # 2️⃣ 의미 기반 정규화
        s = s.replace({
            "nan": "not_applicable",
            "n/a": "not_applicable",
            "na": "not_applicable",
            "unknown": "not_applicable",
            "": "not_applicable",

            "none": "none",

            "no": "not_applicable",
            "not pregnant": "not_applicable",
            "not_pregnant": "not_applicable",

            "yes": "pregnant_normal",
            "pregnant": "pregnant_normal",
        })

        # 3️⃣ ⭐ 반드시 ENCODING_MAP key와 정확히 맞춤
        s = s.replace({
            "not_applicable": "Not_Applicable",
            "none": "None",
            "pregnant_normal": "Pregnant_Normal",
        })

        # 4️⃣ 그래도 혹시 남은 값 → 전부 Not_Applicable
        valid_keys = set(ENCODING_MAP["Pregnancy"].keys())
        s = s.where(s.isin(valid_keys), "Not_Applicable")

        df["Pregnancy"] = s

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, mapping in ENCODING_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

            # ✅ 컬럼별 안전한 기본값 선택
            # - Not_Applicable 있으면 그 값
            # - 없으면 첫 번째 클래스 값
            default_value = (
                mapping["Not_Applicable"]
                if "Not_Applicable" in mapping
                else list(mapping.values())[0]
            )

            df[col] = df[col].fillna(default_value)

    return df



def build_features(df: pd.DataFrame, mode: str):
    df = df.copy()
    mode = mode.lower()

    # ✅ 반드시 이 순서
    df = normalize_categorical_values(df)
    df = encode_categorical_features(df)

    if mode == "sg_only":
        feature_cols = ["SG"]
    elif mode == "sg_plus_meta":
        feature_cols = [
            "SG",
            "Meal_Status",
            "BMI_Class",
            "Age_Group",
            "Exercise",
            "Family_History",
            "Pregnancy"
        ]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    X = df[feature_cols].values
    feature_names = feature_cols

    y = df["BG"].values if "BG" in df.columns else None

    return X, y, feature_names
