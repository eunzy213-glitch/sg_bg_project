# main.py
# ============================================================
# 이 파일은 프로젝트의 실행 시작점입니다.
# 여기서 어떤 실험을 돌릴지 정의합니다.
# ============================================================

# ---------------------------------------------------------
# 🔒 Global Random Seed Fix (재현성 보장)
# ---------------------------------------------------------
import os
import random
import numpy as np

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED) # 파이썬 내부 해시 연산의 랜덤성 고정
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# ✅ Logger 설정
# ---------------------------------------------------------
from src.logger import setup_logging, get_logger

# 🔥 반드시 logger 생성 전에 한 번만 호출
setup_logging()  

from pipelines.pipeline import run_pipeline   # 전체 학습 파이프라인 실행 함수


# main 전용 로그
logger = get_logger(__name__)  # 🔹 log_file 지정 안 해도 됨 (project.log로 통합됨)

# Python 파일을 직접 실행했을 때만 아래 코드가 실행되도록 하는 표준 구조
if __name__ == "__main__":

    logger.info("🚀 Project execution started")   # 실행 시작 로그

    # 사용할 데이터 파일 경로 (data 폴더 기준)
    data_path = "data/dataset.csv"
    logger.info(f"📂 Using dataset: {data_path}")  # 사용 데이터 로그

    try:
        # --------------------------------------------------------
        # 실험 1️⃣ : SG 단일 변수 실험
        # --------------------------------------------------------
        logger.info("🧪 Experiment 1 started: SG_ONLY")

        run_pipeline(
            data_path=data_path,        # 데이터 경로
            experiment_name="SG_ONLY",   # 결과 저장 폴더 이름
            feature_mode="sg_only"       # feature 구성 방식
        )


        # --------------------------------------------------------
        # 실험 2️⃣ : SG + 추가 상태 변수 실험
        # --------------------------------------------------------
        logger.info("🧪 Experiment 2 started: SG_PLUS_META")

        run_pipeline(
            data_path=data_path,
            experiment_name="SG_PLUS_META",
            feature_mode="sg_plus_meta"
        )


        # --------------------------------------------------------
        # 전체 실행 종료
        # --------------------------------------------------------
        logger.info("🎉 All experiments completed successfully")

    except Exception as e:
        # --------------------------------------------------------
        # ❌ 예외 발생 시 로그 기록
        # --------------------------------------------------------
        logger.exception("🔥 Fatal error occurred during project execution")
        raise e
