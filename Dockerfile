# ================================
# SG → BG Prediction Project
# Default: run training pipeline
# ================================

FROM python:3.10-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 (LightGBM 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 프로젝트 코드 복사
COPY . /app

# 기본 실행: main.py
CMD ["python", "main.py"]
