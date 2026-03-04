#!/usr/bin/env bash
# deploy-rag-bench.sh
# RAG 벤치마크 K8s 빌드 & 배포 헬퍼.
#
# 사용법:
#   source .env && bash k8s/deploy-rag-bench.sh [STEP]
#
# STEP (기본: all):
#   build     — K8s 빌더로 worker 이미지 빌드 & Harbor 푸시
#   secrets   — bench-secrets Secret 생성/갱신
#   infra     — PVC(bench-results, model-cache) + namespace 적용
#   run       — orchestrator.py 실행 (전체 카테고리 × full preset)
#   all       — build → secrets → infra → run 순서로 전체 실행
#
# 환경변수:
#   HARBOR_REGISTRY   Harbor 레지스트리 주소 (필수)
#   HARBOR_USER       Harbor 사용자명 (필수, docker login용)
#   HARBOR_CLI_SECRET Harbor 패스워드/토큰 (필수, docker login용)
#   OPENAI_API_KEY    OpenAI API 키 (선택, secrets 스텝)
#   UPSTAGE_API_KEY   Upstage API 키 (선택, secrets 스텝)
#   NAMESPACE         K8s 네임스페이스 (기본: rag-bench-test)
#   TAG               이미지 태그 (기본: latest)
#   K8S_BUILDER       buildx 빌더 이름 (기본: k8s-amd64)
#   NODE_SELECTOR     K8s 빌더 노드 셀렉터 (기본: management 노드)
set -euo pipefail

# ── 설정 ──────────────────────────────────────────────────────────────────────
NAMESPACE="${NAMESPACE:-rag-bench-test}"
TAG="${TAG:-latest}"
K8S_BUILDER="${K8S_BUILDER:-k8s-amd64}"
NODE_SELECTOR="${NODE_SELECTOR:-node-role.kubernetes.io/management=management}"
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

# 필수 환경변수 확인
require_env HARBOR_REGISTRY

IMAGE_WORKER="$HARBOR_REGISTRY/rag-bench-test/worker:$TAG"

# ── K8s 빌더 헬퍼 ────────────────────────────────────────────────────────────
setup_builder() {
    if docker buildx inspect "$K8S_BUILDER" &>/dev/null; then
        info "빌더 '$K8S_BUILDER' 이미 존재 — 재사용"
    else
        info "K8s 빌더 생성: $K8S_BUILDER"
        docker buildx create \
            --name "$K8S_BUILDER" \
            --driver kubernetes \
            --driver-opt "namespace=$NAMESPACE,nodeselector=$NODE_SELECTOR" \
            --platform linux/amd64
        info "빌더 부트스트랩 중..."
        docker buildx inspect "$K8S_BUILDER" --bootstrap
        ok "빌더 준비 완료"
    fi
}

teardown_builder() {
    info "빌더 제거: $K8S_BUILDER (클러스터 리소스 해제)"
    docker buildx rm "$K8S_BUILDER" || true
}

harbor_login() {
    require_env HARBOR_USER
    require_env HARBOR_CLI_SECRET
    info "Harbor 로그인: $HARBOR_REGISTRY"
    echo "${HARBOR_CLI_SECRET}" \
        | docker login "$HARBOR_REGISTRY" -u "$HARBOR_USER" --password-stdin
    ok "Harbor 로그인 완료"
}

# ── kubectl 헬퍼 ──────────────────────────────────────────────────────────────
kube() { kubectl -n "$NAMESPACE" "$@"; }

# ── STEP: build ──────────────────────────────────────────────────────────────
build_image() {
    info "=== Worker 이미지 빌드 시작 (K8s 빌더: $K8S_BUILDER) ==="
    cd "$REPO_ROOT"

    harbor_login
    setup_builder

    trap 'teardown_builder' EXIT

    info "빌드: $IMAGE_WORKER"
    docker buildx build \
        --builder "$K8S_BUILDER" \
        --platform linux/amd64 \
        --push \
        -t "$IMAGE_WORKER" \
        -f deploy/k8s/dockerfiles/Dockerfile \
        .
    ok "worker 이미지 완료"

    trap - EXIT
    teardown_builder

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

# ── STEP: infra ──────────────────────────────────────────────────────────────
setup_infra() {
    info "=== 인프라 리소스 적용 ==="

    info "네임스페이스 적용..."
    kubectl apply -f "$MANIFESTS/namespace.yaml"

    info "PVC 적용..."
    kube apply -f "$MANIFESTS/results-pvc.yaml"
    kube apply -f "$MANIFESTS/model-cache-pvc.yaml"

    ok "인프라 리소스 적용 완료"
    kube get pvc
}

# ── STEP: run ────────────────────────────────────────────────────────────────
run_bench() {
    require_env HARBOR_REGISTRY

    local categories="general,legal,business,medical,technical"
    local preset="full"

    info "=== 벤치마크 실행: categories=$categories, preset=$preset ==="
    cd "$REPO_ROOT"

    python3 k8s/orchestrator.py \
        --image "$IMAGE_WORKER" \
        --categories "$categories" \
        --preset "$preset"
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
case "$STEP" in
    build)
        build_image
        ;;
    secrets)
        create_secrets
        ;;
    infra)
        setup_infra
        ;;
    run)
        run_bench
        ;;
    all)
        build_image
        create_secrets
        setup_infra
        run_bench
        ;;
    *)
        cat <<EOF
사용법: bash k8s/deploy-rag-bench.sh [STEP]

STEP:
  build     — K8s 빌더로 worker 이미지 빌드 & Harbor 푸시
  secrets   — API 키 Secret 생성/갱신 (OPENAI_API_KEY, UPSTAGE_API_KEY)
  infra     — 네임스페이스 + PVC (bench-results, model-cache) 적용
  run       — orchestrator.py 실행 (5 카테고리 × full preset = 100 조합)
  all       — build → secrets → infra → run 순서 전체 실행

환경변수 (source .env로 로드):
  HARBOR_REGISTRY=harbor.example.com  (필수)
  HARBOR_USER=...                     (필수, docker login용)
  HARBOR_CLI_SECRET=...               (필수, docker login용)
  OPENAI_API_KEY=...                  (선택)
  UPSTAGE_API_KEY=...                 (선택)
  NAMESPACE=rag-bench-test            (기본값)
  TAG=latest                          (기본값)
  K8S_BUILDER=k8s-amd64              (기본값)

예시:
  source .env
  bash k8s/deploy-rag-bench.sh all
  bash k8s/deploy-rag-bench.sh build    # 이미지만 빌드
  bash k8s/deploy-rag-bench.sh run      # 벤치마크만 실행
EOF
        exit 1
        ;;
esac
