# K8s 벤치마크 테스트 결과 및 구성 제안

> 작성일: 2026-02-26
> 클러스터: zcp-ags-cp-eks (ap-northeast-2)

---

## 1. 테스트 결과 요약

### 성공

| 항목 | 설정 | 결과 |
|------|------|------|
| Prep Job (Phase 1) | CPU req/lim: 1/1, Mem: 4Gi/8Gi | 61초 완료 (1000 corpus, 20 QA, 57 chunks enriched) |
| Bench Job (Phase 2) | CPU req/lim: 1/2, Mem: 4Gi/8Gi, TEI 없음 | 정상 동작 — 모델 로딩 168초 (ColBERT 포함) |
| ai-platform toleration | `ai-platform=true:NoSchedule` | 정상 스케줄링 — r5a.xlarge 노드 활용 |
| PVC (EFS RWX) | bench-results 10Gi, model-cache 50Gi | 정상 마운트/읽기/쓰기 |
| harbor-cred ImagePullSecret | ags-registry.ags.cloudzcp.net | 이미지 정상 풀 |

### 실패

| 항목 | 설정 | 원인 | 상세 |
|------|------|------|------|
| TEI bge-m3 (8Gi limit) | memory limit 8Gi | **OOMKilled** | bge-m3 568M param, float32 ONNX warm-up 시 8Gi 초과 |
| TEI bge-m3 (12Gi limit) | memory limit 12Gi | **OOMKilled** | warm-up 단계에서 12Gi도 초과. float32 ONNX 전체 로드 + inference buffer 필요 |
| TEI kosimcse (8Gi limit) | memory limit 8Gi | **SIGSEGV (exit 139)** | Intel MKL SGEMM 파라미터 오류. r5a.xlarge (AMD EPYC) CPU와 TEI cpu-1.7 이미지 비호환 |

### TEI 실패 근본 원인

1. **메모리**: TEI는 ONNX 모델을 float32로 메모리에 전부 로드 + warm-up 배치 처리. bge-m3(568M param × 4bytes = ~2.3GB 가중치) + ONNX 런타임 오버헤드 + warm-up 텐서 → 12Gi 초과.
2. **CPU 호환**: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7`은 Intel AVX 명령어 최적화. AMD EPYC(r5a.xlarge)에서 MKL SGEMM 크래시 발생.

---

## 2. 클러스터 자원 현황

### 노드 (5대)

| 노드 | 인스턴스 | CPU Alloc. | Role | Taint | 비고 |
|------|----------|------------|------|-------|------|
| 0-149 | r5a.xlarge (4C/32G) | 3920m | ai-platform | `NoSchedule` | AMD EPYC |
| 0-175 | r5a.xlarge (4C/32G) | 3920m | ai-platform | `NoSchedule` | AMD EPYC |
| 0-188 | m7i.2xlarge (8C/32G) | 7910m | logging,mgmt | 없음 | Intel, 59% 점유 |
| 1-173 | m8i.2xlarge (8C/32G) | 7910m | edge,monitoring | 없음 | Intel, 86% 점유 |
| 1-184 | m8i.2xlarge (8C/32G) | 7910m | logging,mgmt | 없음 | Intel, 62% 점유 |

### 가용 CPU (request 기준)

| 구분 | 가용 CPU | 비고 |
|------|----------|------|
| taint 없는 노드 (#3,4,5) | ~7.2 vCPU | #4는 1vCPU만 여유 |
| ai-platform 노드 (#1,2) | ~5.8 vCPU | toleration 추가 시 활용 가능 |
| **합계** | **~13 vCPU** | |

---

## 3. 벤치마크 구성 제안

### 방안 A: TEI 없이 직접 임베딩 (권장)

워커 Pod 내에서 HuggingFace 모델을 직접 로딩하는 기존 방식.

```bash
python3 k8s/orchestrator.py \
  --image $IMAGE \
  --categories general \
  --preset service \
  --bench-cpu-request 1 --bench-cpu-limit 2 \
  --bench-memory-request 4Gi --bench-memory-limit 8Gi \
  --max-corpus 1000 --max-queries 50
```

**장점:**
- TEI 호환 문제 없음
- 추가 인프라 불필요
- model-cache PVC로 모델 공유 (첫 Job만 다운로드)

**단점:**
- Job마다 모델 로딩 시간 (30초~2분)
- Job당 메모리 사용 증가 (kosimcse ~440MB, bge-m3 ~2.3GB)

**리소스 설정:**

| 모델 | 워커 Memory limit 권장 |
|------|----------------------|
| kosimcse (768d, 110M) | 6Gi |
| e5-large (1024d, 560M) | 8Gi |
| bge-m3 (1024d, 568M) | 8Gi |
| + ColBERT reranker | +1Gi |
| + SPLADE sparse | +1Gi |

→ 최악 케이스 (bge-m3 + SPLADE + ColBERT): **~6GB RAM**, 8Gi limit이면 충분.

**동시 실행 수:**
- CPU req 1 기준: ~13개 동시 (5노드)
- service 프리셋: 4 cat × 6 combo = 24 → 13개 동시 + 11개 순차 대기
- 예상 소요: Prep ~5분 + Bench ~40분 (소량 데이터 기준)

### 방안 B: TEI + Intel 노드 전용 스케줄링

TEI를 Intel 노드(m7i/m8i)에만 배치하고, nodeSelector로 AMD 노드를 회피.

```yaml
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: m7i.2xlarge  # 또는 m8i.2xlarge
  containers:
    - name: tei
      resources:
        limits:
          memory: "16Gi"  # bge-m3 float32 기준
```

**장점:**
- 모델 로딩 1회 (모든 Job이 공유)
- 워커 Pod 메모리 경감

**단점:**
- Intel 노드 여유가 적음 (~7.2 vCPU, bge-m3 TEI만 16Gi 필요)
- TEI 3개 모델 × 16Gi = 48Gi → management 노드 메모리 초과 위험
- 관리 복잡도 증가

### 방안 C: TEI + GPU 노드 추가 (최적)

GPU 노드를 추가하여 TEI를 CUDA 모드로 실행. 모든 모델 동시 서빙 가능.

```yaml
image: ghcr.io/huggingface/text-embeddings-inference:1.7  # GPU 버전
args: ["--model-id", "BAAI/bge-m3", "--dtype", "float16"]
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "8Gi"
```

**장점:**
- 가장 빠른 임베딩 처리
- float16으로 메모리 절감
- CPU/AMD 호환 문제 없음

**단점:**
- 추가 GPU 노드 비용 (g4dn.xlarge ~$0.53/hr)
- GPU 노드 프로비저닝 시간

---

## 4. 최종 권장

| 우선순위 | 방안 | 사유 |
|---------|------|------|
| **1순위** | **방안 A (TEI 없이 직접 임베딩)** | 즉시 실행 가능, 추가 인프라 불필요 |
| 2순위 | 방안 B (TEI + Intel 전용) | kosimcse/e5만 TEI, bge-m3는 직접 임베딩 하이브리드 |
| 3순위 | 방안 C (GPU 노드) | 프로덕션급 벤치마크 시 |

### 즉시 실행 가능한 명령어

```bash
# 소량 테스트 (general 카테고리, 1000 corpus, 50 QA)
python3 k8s/orchestrator.py \
  --image ags-registry.ags.cloudzcp.net/rag-bench-test/worker:latest \
  --categories general \
  --preset service \
  --max-corpus 1000 --max-queries 50

# 전체 벤치마크 (4 카테고리)
python3 k8s/orchestrator.py \
  --image ags-registry.ags.cloudzcp.net/rag-bench-test/worker:latest \
  --categories general,legal,business,medical \
  --preset service \
  --max-corpus 10000 --max-queries 100
```

---

## 5. 코드 변경 이력

| 파일 | 변경 |
|------|------|
| `k8s/manifests/bench-job-template.yaml` | ai-platform toleration 추가 |
| `k8s/manifests/prep-job-template.yaml` | ai-platform toleration 추가 |
| `k8s/manifests/tei-deployment.yaml` | ai-platform toleration 추가, memory 6Gi/12Gi |
| `k8s/orchestrator.py` | Prep CPU 1/1, Bench CPU 1/2, Mem 4Gi/8Gi (방안 A 기본값) |
| `k8s/worker_entrypoint.py` | COMBO_LABEL fallback 제거 (필수 env로 변경) |
