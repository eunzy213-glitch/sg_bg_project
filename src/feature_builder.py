# preprocessing.py
# ============================================================
# DataFrame 타입 힌트 및 데이터 처리용
# ============================================================

import pandas as pd # DataFrame 처리 라이브러리

# 각 범주형 컬럼을 정수로 바꿀 때 사용한 매핑표
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
        "Not_Pregnant": 0,      
        "Pregnant_Normal": 1,   
        "Pregnant_GDM": 2       
    }
}


def normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 값이 데이터에 다양한 표기로 들어와도,
    ENCODING_MAP의 key로 강제 통일(정규화)합니다.
    """

    df = df.copy()  # 원본 보호(함수 밖 df가 변하지 않게)

    # --------------------------------------------------------
    # ✅ Pregnancy 값 정규화 (가장 중요/가장 오류 잦음)
    # --------------------------------------------------------
    if "Pregnancy" in df.columns:

        # 1) 결측/공백/대소문자/앞뒤공백 처리 통일
        s = (
            df["Pregnancy"]
            .fillna("not_pregnant")      # NaN이면 기본적으로 임신 아님으로 처리
            .astype(str)                 # 문자열로 통일
            .str.strip()                 # 앞뒤 공백 제거
            .str.lower()                 # 소문자 통일
        )

        # 2) 데이터에 존재 가능한 다양한 표현을 의미 기반으로 통일
        s = s.replace({
            # ----- "임신 아님" 계열 -----
            "nan": "not_pregnant",
            "n/a": "not_pregnant",
            "na": "not_pregnant",
            "unknown": "not_pregnant",
            "": "not_pregnant",

            "none": "not_pregnant",            
            "not_applicable": "not_pregnant",  
            "not applicable": "not_pregnant",
            "no": "not_pregnant",
            "not pregnant": "not_pregnant",
            "not_pregnant": "not_pregnant",

            # ----- "정상 임신" 계열 -----
            "pregnant_normal": "pregnant_normal",
            "pregnant normal": "pregnant_normal",
            "pregnant": "pregnant_normal",     # 임신이라고만 쓰인 경우 우선 정상 임신으로 간주
            
            # ----- "임신성 당뇨(GDM)" 계열 -----
            "pregnant_gdm": "pregnant_gdm",
            "pregnant gdm": "pregnant_gdm",
            "gdm": "pregnant_gdm",
            "gestational diabetes": "pregnant_gdm",
            "gestational_diabetes": "pregnant_gdm",
        })

        # 3) 최종적으로 ENCODING_MAP의 key와 정확히 맞추기(대소문자 포함)
        s = s.replace({
            "not_pregnant": "Not_Pregnant",
            "pregnant_normal": "Pregnant_Normal",
            "pregnant_gdm": "Pregnant_GDM",
        })

        # 4) 그래도 남는 이상값은 임신 아님으로 강제(안정성 우선)
        valid_keys = set(ENCODING_MAP["Pregnancy"].keys())  # 허용 키 집합
        s = s.where(s.isin(valid_keys), "Not_Pregnant")     # 허용되지 않으면 Not_Pregnant로

        # 5) 정규화 결과를 df에 반영
        df["Pregnancy"] = s

    return df  # 정규화된 df 반환


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENCODING_MAP 기준으로 범주형 값을 정수로 변환합니다.
    학습/추론/시각화 모두 동일한 인코딩 규칙을 사용해야 합니다.
    """

    df = df.copy()  # 원본 보호

    for col, mapping in ENCODING_MAP.items():  # 각 컬럼별 매핑 적용
        if col in df.columns:                  # 실제 df에 컬럼이 있을 때만
            df[col] = df[col].map(mapping)     # 문자열 -> 정수

            # 기본값 정책:
            # - Pregnancy는 Not_Pregnant가 기본값
            # - 그 외는 첫 번째 클래스 값
            default_value = (
                mapping["Not_Pregnant"]
                if col == "Pregnancy"
                else list(mapping.values())[0]
            )

            df[col] = df[col].fillna(default_value)  # NaN이면 기본값으로 채우기

    return df  # 인코딩된 df 반환


def build_features(df: pd.DataFrame, mode: str):
    """
    mode에 따라 입력 feature 구성을 결정하고,
    X(입력), y(정답), feature_names(컬럼명 리스트)를 반환합니다.
    """

    df = df.copy()            # 원본 보호
    mode = mode.lower()       # 비교 안정성 확보

    # ✅ 반드시 정규화 -> 인코딩 순서
    df = normalize_categorical_values(df)  # 표기 통일
    df = encode_categorical_features(df)   # 정수 인코딩

    # 모드별 feature 구성
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
            "Pregnancy",  
        ]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    X = df[feature_cols].values             # 모델 입력 행렬
    feature_names = feature_cols            # feature 이름(순서 중요)
    y = df["BG"].values if "BG" in df.columns else None  # 정답(추론이면 None)

    return X, y, feature_names
