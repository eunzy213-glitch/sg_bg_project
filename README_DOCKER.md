# 🐳 SG → BG 예측 프로젝트 Docker 실행 가이드

본 문서는 **SG → BG 예측 프로젝트를 Docker 환경에서 실행하기 위한 전용 가이드**입니다.
Docker를 사용하면 로컬 환경과 무관하게 **동일한 실행 환경에서 학습·실험·추론을 재현**할 수 있습니다.

---
## 📌 Docker 사용 목적

이 프로젝트에서 Docker는 다음을 목적으로 사용됩니다.

 - 실험 환경 재현성(Reproducibility) 확보
 - Python / 라이브러리 버전 충돌 방지
 - 학습 파이프라인을 단일 명령으로 실행
 - Streamlit / CLI 추론 환경을 손쉽게 실행

---

## 📁 Docker 관련 파일 구성
```text
sg_bg_project/
├── Dockerfile          # Docker 이미지 빌드 설정
├── .dockerignore       # Docker 빌드 제외 파일 목록
├── README_DOCKER.md    # Docker 실행 가이드 (본 문서)
```

---

## 🧱 Docker 이미지 빌드
프로젝트 루트 디렉토리에서 아래 명령을 실행합니다.
```bash
docker build -t sg-bg .
```
 - `sg-bg` : 생성할 Docker 이미지 이름
 - 최초 빌드 시 라이브러리 설치로 인해 시간이 소요될 수 있습니다.

---

## 🏃 기본 실행 (학습 파이프라인)
Docker 컨테이너를 실행하면 기본적으로 `main.py`가 실행되며 학습 파이프라인이 수행됩니다.
```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  sg-bg \
  python main.py
```
실행 내용:
  - 데이터 로드
  - 전처리 및 이상치 제거
  - Feature 생성
  - 모델 학습 및 평가
  - 결과 CSV / 시각화 이미지 생성
  - 추론용 최적 모델(`.pkl`) 저장

  📌 **Docker의 기본 실행 목적은 학습 파이프라인 재현입니다.**

  ---

## 🔍 설명가능성 파이프라인 실행 (선택) 
SHAP / LIME 기반의 모델 해석이 필요한 경우 아래와 같이 실행할 수 있습니다.

```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  sg-bg \
  python -m pipelines.explain_pipeline
```

실행 결과:
 - SHAP summary plot
 - LIME local explanation plot
 - `results/*/EXPLAIN_모델명/` 폴더에 저장

---

## 🤖 CLI 기반 추론 실행
학습된 모델을 이용해 **새로운 입력값으로 BG를 예측**할 수 있습니다.

```bash
docker run --rm -it \
  -v $(pwd)/results:/app/results \
  sg-bg \
  python -m inference.cli_predict
```

CLI에서 다음 정보를 순서대로 입력합니다.
 - SG(Salivary Glucose)
 - Meal_Status
 - BMI_Class
 - Age_Group
 - Exercise
 - Family_History
 - Pregnancy

---

## 🌐 Streamlit 웹 인터페이스 실행 (선택)
웹 UI 기반 예측 및 시각화를 실행하려면 다음 명령을 사용합니다.

```bash
docker run --rm -it \
  -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  sg-bg \
  streamlit run /app/app/interactive_app.py
```

웹 브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

---

## 📌 주의 사항
 - `data/`, `results/`, `*.pkl` 파일은 Docker 이미지 및 GitHub 저장소에 포함되지 않습니다.
 - Docker 컨테이너 실행 전, 필요한 데이터는 로컬 환경에 준비되어 있어야 합니다.
 - 추론 파이프라인은 학습과 동일한 feature encoding 로직을 사용합니다.

---

## 🔚 정리
 - Docker 기본 실행 → 학습 파이프라인
 - 필요에 따라
   - 설명가능성 파이프라인 실행
   - CLI 추론 실행
   - Streamlit 웹 실행

👉 **Docker는 이 프로젝트의 실행 환경을 표준화하는 도구**입니다.

---

## 📎 참고

프로젝트 전체 설명 및 구조는 👉 `README.md` 파일을 참고하세요.
