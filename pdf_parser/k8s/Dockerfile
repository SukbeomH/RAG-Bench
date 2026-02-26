# syntax=docker/dockerfile:1
# PDF Parser Worker
# PDF → Markdown 변환을 K8s Job으로 실행하기 위한 워커 이미지.
#
# Build (프로젝트 루트에서):
#   docker buildx build --platform linux/amd64 -t $REGISTRY/pdf-parser/worker:latest -f pdf_parser/k8s/Dockerfile --load .

# ── Stage 1: 빌드 ──
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 설치
COPY pdf_parser/k8s/requirements-worker.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Stage 2: 런타임 ──
FROM python:3.12-slim AS runtime

# Docling OCR 런타임 의존성 (tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-kor \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# pdf_parser 모듈 복사
COPY pdf_parser/*.py ./pdf_parser/
COPY pdf_parser/k8s/worker_entrypoint.py ./

ENTRYPOINT ["python", "-u", "worker_entrypoint.py"]
