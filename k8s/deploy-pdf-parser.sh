#!/usr/bin/env bash
# deploy-pdf-parser.sh
# PDF Parser K8s 벤치마크 인프라 빌드 & 배포 헬퍼.
#
# 사용법:
#   export HARBOR_REGISTRY=harbor.example.com   # 또는 .env에서 로드
#   source .env && bash k8s/deploy-pdf-parser.sh [STEP]
#
# STEP (기본: all):
#   images    — Docker 이미지 빌드 & 푸시 (pdf-parser, got-ocr2, paddleocr-vl)
#   services  — 오픈소스 VLM 서비스 배포 (Ollama, GOT-OCR2, PaddleOCR-VL)
#   secrets   — API 키 Secret 생성/갱신
#   all       — images → secrets → services 순서로 전체 실행
#
# 환경변수:
#   HARBOR_REGISTRY   Harbor 레지스트리 주소 (필수)
#   OPENAI_API_KEY    OpenAI API 키 (선택, secrets 스텝)
#   UPSTAGE_API_KEY   Upstage API 키 (선택, secrets 스텝)
#   NAMESPACE         K8s 네임스페이스 (기본: rag-bench-test)
#   TAG               이미지 태그 (기본: latest)
#   PUSH              이미지 푸시 여부 (기본: true, false로 로컬 테스트)
set -euo pipefail

# ── 설정 ──────────────────────────────────────────────────────────────────────
NAMESPACE="${NAMESPACE:-rag-bench-test}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-true}"
STEP="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFESTS="$SCRIPT_DIR/manifests"

# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { log "INFO  $*"; }
ok()   { log "OK    $*"; }
fail() { log "ERROR $*" >&2; exit 1; }

require_env() {
    [[ -n "${!1:-}" ]] || fail "환경변수 $1 미설정. .env 로드 또는 export $1=... 필요"
}

# Harbor 레지스트리 필수 확인
require_env HARBOR_REGISTRY

IMAGE_PDF_PARSER="$HARBOR_REGISTRY/rag-bench-test/pdf-parser:$TAG"
IMAGE_GOT_OCR2="$HARBOR_REGISTRY/rag-bench-test/got-ocr2:$TAG"
IMAGE_PADDLEOCR_VL="$HARBOR_REGISTRY/rag-bench-test/paddleocr-vl:$TAG"

# ── kubectl 헬퍼 ──────────────────────────────────────────────────────────────
kube() { kubectl -n "$NAMESPACE" "$@"; }

apply_manifest() {
    local yaml_file="$1"
    # ${IMAGE_REGISTRY} 플레이스홀더 치환
    sed "s|\${IMAGE_REGISTRY}|$HARBOR_REGISTRY|g" "$yaml_file" \
        | kubectl -n "$NAMESPACE" apply -f -
}

# ── STEP: images ──────────────────────────────────────────────────────────────
build_images() {
    info "=== Docker 이미지 빌드 시작 ==="
    cd "$REPO_ROOT"

    local push_flag="--load"
    [[ "$PUSH" == "true" ]] && push_flag="--push"

    # pdf-parser 워커 이미지
    info "빌드: $IMAGE_PDF_PARSER"
    docker buildx build \
        --platform linux/amd64 \
        -t "$IMAGE_PDF_PARSER" \
        -f k8s/Dockerfile.pdf-parser \
        $push_flag \
        .
    ok "pdf-parser 이미지 완료"

    # GOT-OCR2 서버 이미지
    info "빌드: $IMAGE_GOT_OCR2"
    docker buildx build \
        --platform linux/amd64 \
        -t "$IMAGE_GOT_OCR2" \
        -f k8s/Dockerfile.got-ocr2 \
        $push_flag \
        .
    ok "got-ocr2 이미지 완료"

    # PaddleOCR-VL 서버 이미지
    info "빌드: $IMAGE_PADDLEOCR_VL"
    docker buildx build \
        --platform linux/amd64 \
        -t "$IMAGE_PADDLEOCR_VL" \
        -f k8s/Dockerfile.paddleocr-vl \
        $push_flag \
        .
    ok "paddleocr-vl 이미지 완료"

    info "=== 이미지 빌드 완료 ==="
}

# ── STEP: secrets ─────────────────────────────────────────────────────────────
create_secrets() {
    info "=== API 키 Secret 생성/갱신 ==="

    local openai_key="${OPENAI_API_KEY:-}"
    local upstage_key="${UPSTAGE_API_KEY:-}"

    if [[ -z "$openai_key" && -z "$upstage_key" ]]; then
        info "SKIP: OPENAI_API_KEY, UPSTAGE_API_KEY 모두 미설정 — Secret 생략"
        return 0
    fi

    kubectl -n "$NAMESPACE" create secret generic bench-secrets \
        --from-literal=OPENAI_API_KEY="${openai_key}" \
        --from-literal=UPSTAGE_API_KEY="${upstage_key}" \
        --dry-run=client -o yaml | kubectl apply -f -

    ok "bench-secrets 갱신 완료"
}

# ── STEP: services ────────────────────────────────────────────────────────────
deploy_services() {
    info "=== 오픈소스 VLM 서비스 배포 ==="

    # 1. Ollama (Granite Vision 2B) — ${IMAGE_REGISTRY} 플레이스홀더 없음
    info "Ollama 배포..."
    kube apply -f "$MANIFESTS/ollama-deployment.yaml"

    # 2. GOT-OCR2 (${IMAGE_REGISTRY} 치환 필요)
    info "GOT-OCR2 배포..."
    apply_manifest "$MANIFESTS/got-ocr2-deployment.yaml"

    # 3. PaddleOCR-VL (${IMAGE_REGISTRY} 치환 필요)
    info "PaddleOCR-VL 배포..."
    apply_manifest "$MANIFESTS/paddleocr-vl-deployment.yaml"

    ok "배포 YAML 적용 완료"
    info ""
    info "서비스 준비까지 대기 (모델 다운로드 포함, 최대 15분)..."

    # Ollama readiness 대기
    info "Ollama 대기 중..."
    kube rollout status deployment/ollama-server --timeout=600s || true

    # GOT-OCR2 readiness 대기
    info "GOT-OCR2 대기 중..."
    kube rollout status deployment/got-ocr2-server --timeout=600s || true

    # PaddleOCR-VL readiness 대기
    info "PaddleOCR-VL 대기 중..."
    kube rollout status deployment/paddleocr-vl-server --timeout=600s || true

    info ""
    info "=== 서비스 상태 ==="
    kube get pods,svc -l component=inference
    info "=== VLM 서비스 배포 완료 ==="
}

# ── STEP: verify ──────────────────────────────────────────────────────────────
verify_services() {
    info "=== 서비스 헬스체크 ==="

    # Ollama
    if kube run verify-ollama --rm -it --restart=Never \
        --image=curlimages/curl:latest \
        --command -- curl -sf http://ollama-server:11434/api/tags 2>/dev/null; then
        ok "Ollama 정상"
    else
        info "WARN: Ollama 응답 없음 (아직 초기화 중일 수 있음)"
    fi

    # GOT-OCR2
    if kube run verify-got --rm -it --restart=Never \
        --image=curlimages/curl:latest \
        --command -- curl -sf http://got-ocr2-server:8000/health 2>/dev/null; then
        ok "GOT-OCR2 정상"
    else
        info "WARN: GOT-OCR2 응답 없음"
    fi

    # PaddleOCR-VL
    if kube run verify-paddle --rm -it --restart=Never \
        --image=curlimages/curl:latest \
        --command -- curl -sf http://paddleocr-vl-server:8000/health 2>/dev/null; then
        ok "PaddleOCR-VL 정상"
    else
        info "WARN: PaddleOCR-VL 응답 없음"
    fi
}

# ── STEP: run-bench ───────────────────────────────────────────────────────────
run_benchmark() {
    local preset="${2:-quick}"
    require_env HARBOR_REGISTRY

    info "=== 벤치마크 실행: preset=$preset ==="
    cd "$REPO_ROOT"

    python k8s/pdf_parser_orchestrator.py \
        --image "$IMAGE_PDF_PARSER" \
        --preset "$preset"
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
case "$STEP" in
    images)
        build_images
        ;;
    secrets)
        create_secrets
        ;;
    services)
        deploy_services
        ;;
    verify)
        verify_services
        ;;
    bench)
        run_benchmark "${@}"
        ;;
    all)
        build_images
        create_secrets
        deploy_services
        ;;
    *)
        cat <<EOF
사용법: bash k8s/deploy-pdf-parser.sh [STEP]

STEP:
  images    — Docker 이미지 빌드 & 푸시
  secrets   — API 키 Secret 생성/갱신
  services  — VLM 서비스 K8s 배포 (Ollama, GOT-OCR2, PaddleOCR-VL)
  verify    — 서비스 헬스체크
  bench [PRESET] — 벤치마크 실행 (기본: quick)
  all       — images → secrets → services 순서 전체 실행

환경변수:
  HARBOR_REGISTRY=harbor.example.com  (필수)
  OPENAI_API_KEY=...                  (선택)
  UPSTAGE_API_KEY=...                 (선택)
  NAMESPACE=rag-bench-test            (기본값)
  TAG=latest                          (기본값)
  PUSH=true                           (false 시 로컬 --load)

예시:
  source .env
  bash k8s/deploy-pdf-parser.sh all
  bash k8s/deploy-pdf-parser.sh bench phase1
  PUSH=false bash k8s/deploy-pdf-parser.sh images
EOF
        exit 1
        ;;
esac
