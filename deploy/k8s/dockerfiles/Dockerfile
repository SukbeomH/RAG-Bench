# syntax=docker/dockerfile:1
# AutoRAG Benchmark Worker
# 카테고리별 벤치마크를 K8s Job으로 실행하기 위한 워커 이미지.
#
# Build (프로젝트 루트에서):
#   docker buildx build --platform linux/amd64 -t $HARBOR_REGISTRY/rag-bench-test/worker:latest -f k8s/Dockerfile --load .

# ── Stage 1: 빌드 (컴파일 의존성 포함) ──
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# venv 생성 (멀티스테이지 복사 안정성 확보)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch CPU-only 먼저 설치 (CUDA 번들 제거 → ~2.5GB 절감)
# --mount=type=cache: pip 다운로드 캐시를 빌드 간 재사용
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치 (requirements 변경 시에만 재실행)
COPY k8s/requirements-worker.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 의존성이 GPU torch를 끌어왔을 경우 CPU-only로 강제 교체 + nvidia 패키지 제거
# 멀티스테이지 빌드이므로 최종 venv 상태만 런타임에 복사됨
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps && \
    pip uninstall -y $(pip list 2>/dev/null | grep -oP '^nvidia-\S+') triton 2>/dev/null; true

# rag_bench 패키지 설치 (소스 변경 시에만 재실행)
COPY pyproject.toml ./
COPY rag_bench/ ./rag_bench/
RUN pip install --no-cache-dir --no-deps .

# ── Stage 2: 런타임 (경량) ──
FROM python:3.12-slim

# KoNLPy → Java (런타임에만 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# venv 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY k8s/worker_entrypoint.py ./worker_entrypoint.py

ENV HF_HOME=/models
ENV TOKENIZERS_PARALLELISM=false
ENV RESULTS_DIR=/results
ENV WORKSPACE_DIR=/workspace

RUN mkdir -p /results /workspace /models

ENTRYPOINT ["python", "-u", "worker_entrypoint.py"]
