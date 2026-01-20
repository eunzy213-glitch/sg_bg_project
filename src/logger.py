# src/logger.py
import logging
import os

def setup_logging(log_dir="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 이미 핸들러가 있으면 중복 생성 방지
    if root_logger.handlers:
        return

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 파일 핸들러 (🔥 모든 모듈 로그가 여기로 모임)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "project.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name):
    return logging.getLogger(name)
