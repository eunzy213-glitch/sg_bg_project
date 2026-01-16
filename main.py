# main.py
# ============================================================
# 이 파일은 프로젝트의 실행 시작점입니다.
# 여기서 어떤 실험을 돌릴지 정의합니다.
# ============================================================

from pipelines.pipeline import run_pipeline # pipelines/pipeline.py 안에 있는 함수를 가져옴


# Python 파일을 직접 실행했을 때만 아래 코드가 실행되도록 하는 표준 구조
if __name__ == "__main__":

    # 사용할 데이터 파일 경로 (data 폴더 기준)
    data_path = "data/dataset.csv"

    # --------------------------------------------------------
    # 실험 1️⃣ : SG 단일 변수 실험
    # --------------------------------------------------------
    run_pipeline(
        data_path=data_path,        # 데이터 경로
        experiment_name="SG_ONLY",   # 결과 저장 폴더 이름
        feature_mode="sg_only"       # feature 구성 방식
    )

    # --------------------------------------------------------
    # 실험 2️⃣ : SG + 추가 상태 변수 실험
    # --------------------------------------------------------
    run_pipeline(
        data_path=data_path,
        experiment_name="SG_PLUS_META",
        feature_mode="sg_plus_meta"
    )
