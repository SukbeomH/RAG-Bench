# K8s 병렬 벤치마크 시스템 — 설계 문서

## 1. 설계 목표

로컬에서 순차 실행하던 RAG 벤치마크를 원격 K8s 클러스터에서 **카테고리 × 전략 조합 단위로 완전 병렬화**하여 총 실행 시간을 최소화한다.

### 제약 조건

| 항목 | 내용 |
|------|------|
| LLM | OpenAI API만 사용 (Ollama 미사용) |
| Qdrant | 파일 모드(`path=`) — 별도 서버 불필요 |
| 병렬 단위 | (카테고리 × 전략 조합) = 최대 24 Jobs (service) / 80 Jobs (full) |
| 중복 방지 | Contextual enrichment(OpenAI 호출)은 카테고리당 1회만 실행 |

---

## 2. 2-Phase 아키텍처

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 orchestrator.py (로컬)               │
                    │  Phase 1 생성 → 대기 → PVC 검증 → Phase 2 생성      │
                    │  → 대기 → 수집 → 병합                               │
                    └───────────┬─────────────────────────────────────────┘
                                │ kubectl apply
                                ▼
┌──────────────── K8s Cluster (rag-bench-test namespace) ─────────────────┐
│                                                                          │
│  Phase 1: Prep (카테고리당 1 Job)                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ prep-general │ │ prep-legal   │ │ prep-business│ │ prep-medical │   │
│  │ HF 로드      │ │ HF 로드      │ │ HF 로드      │ │ HF 로드      │   │
│  │ 청킹         │ │ 청킹         │ │ 청킹         │ │ 청킹         │   │
│  │ Enrichment   │ │ Enrichment   │ │ Enrichment   │ │ Enrichment   │   │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│  ┌─────────────────── PVC: bench-results ──────────────────────────┐   │
│  │ /results/<category>/prepared/                                    │   │
│  │   child_chunks.json, parent_pairs.json, qa_pairs.json           │   │
│  │   enriched_chunks.json, DONE                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                              │
│         │  PVC 가시성 검증 (verify Pod)                                 │
│         ▼                                                              │
│  Phase 2: Bench (카테고리 × 조합별 1 Job)                               │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐       │
│  │ general/         │ │ general/         │ │ general/         │       │
│  │ bgem3-koreanbm25 │ │ bgem3-splade    │ │ e5-koreanbm25   │  ...  │
│  │ 인덱스 빌드       │ │ 인덱스 빌드      │ │ 인덱스 빌드      │       │
│  │ Pass1 + Pass2    │ │ Pass1 + Pass2    │ │ Pass1 + Pass2    │       │
│  └──────┬───────────┘ └──────┬───────────┘ └──────┬───────────┘       │
│         │  (×4 카테고리 = 최대 24개 동시)            │                    │
│         ▼                                           ▼                   │
│  ┌─────────────────── PVC: bench-results ──────────────────────────┐   │
│  │ /results/<category>/<combo_label>/                               │   │
│  │   result.json, latency.csv, ragas.csv, DONE                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────── PVC: model-cache ────────────────────────────┐   │
│  │ HuggingFace 모델 공유 캐시 (중복 다운로드 방지)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Phase 분리 근거

| 작업 | 비용 | 공유 범위 | Phase |
|------|------|-----------|-------|
| HF 데이터 로드 + 청킹 | 중 (네트워크 I/O) | 카테고리 내 모든 조합 | 1 |
| Contextual enrichment | **높음** (OpenAI API 비용) | 카테고리 내 모든 조합 | 1 |
| 인덱스 빌드 (Qdrant) | 중 (CPU) | 조합별 독립 | 2 |
| Pass 1 레이턴시 | 낮음 | 조합별 독립 | 2 |
| Pass 2 RAGAS 평가 | 중 (OpenAI API) | 조합별 독립 | 2 |

Phase 1에서 enrichment를 한 번만 실행하고 직렬화하면, Phase 2의 모든 Job이 `pre_enriched` 파라미터로 주입받아 **LLM 호출 없이** 인덱스를 빌드한다.

---

## 3. 데이터 흐름

### 3.1 직렬화 형식

모든 중간 데이터는 JSON으로 직렬화된다.

```
/results/<category>/prepared/
├── child_chunks.json      # [{page_content, metadata}]
├── parent_pairs.json      # [{parent_id, page_content, metadata}]
├── qa_pairs.json          # [{question, ground_truth, ...}]
├── enriched_chunks.json   # [{page_content: "prefix\n\noriginal", metadata: {..., contextual_prefix, original_content}}]
└── DONE                   # 완료 시그널 (메타데이터 JSON)
```

### 3.2 pre_enriched 경로

```
Phase 1: enrich_only(child_chunks) → enriched_chunks.json (PVC 저장)
                                          │
Phase 2: deserialize → enriched_chunks ──→ build_strategy_from_spec(pre_enriched=enriched_chunks)
                                                    │
                                          cache.py:get_or_build_contextual()
                                                    │
                                          ctx_base.index(pre_enriched)  ← LLM 호출 없음
```

### 3.3 결과 구조

```
/results/<category>/<combo_label>/
├── result.json            # {category, combo, n_qa, ragas: [...], timestamp}
├── latency.csv            # strategy, query, latency_ms, ...
├── ragas.csv              # strategy, faithfulness, answer_relevancy, ...
└── DONE                   # 완료 시그널 — 반드시 마지막에 기록
```

---

## 4. 안전성 설계

### 4.1 원자적 파일 쓰기

동시에 여러 Job이 같은 PVC에 쓰므로, 모든 파일 쓰기는 원자적으로 수행한다.

```python
def _write_json(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.rename(path)  # POSIX rename = atomic
```

CSV도 동일한 패턴을 사용한다. `DONE` 파일은 반드시 모든 데이터 파일 기록 **후** 마지막에 생성하여, 수집기가 불완전한 데이터를 읽는 것을 방지한다.

### 4.2 Phase 1→2 PVC 가시성 검증

ReadWriteMany PVC(NFS/CephFS)는 한 노드의 쓰기가 다른 노드에서 즉시 보이지 않을 수 있다. Phase 1 완료 후 **검증 Pod**를 생성하여 모든 카테고리의 `prepared/DONE` 파일이 읽을 수 있는지 확인한 후에만 Phase 2를 시작한다.

```
Phase 1 완료 → 검증 Pod(sleep 5 && test -f ...) → Phase 2 시작
```

### 4.3 라벨 경로 일치 보장

오케스트레이터와 워커 간 결과 디렉토리 경로 불일치를 방지하기 위해, **오케스트레이터가 생성한 `COMBO_LABEL` 환경변수**를 워커가 그대로 사용한다. 워커는 자체적으로 라벨을 재구성하지 않는다.

```
orchestrator._safe_label(combo)  →  COMBO_LABEL 환경변수  →  worker 결과 디렉토리명
         (단일 진실 공급원)
```

### 4.4 부분 실패 감지

일부 Phase 2 Job이 실패해도 나머지 결과를 수집/병합할 수 있다. 단, 병합 시 기대 조합 수와 실제 조합 수를 비교하여 누락된 조합을 명시적으로 경고한다.

```
WARN [general] 4/6 조합만 성공 — 누락: [e5+splade+colbert, kosimcse+splade+colbert]
```

---

## 5. 빌드 플랫폼

K8s 클러스터는 amd64 아키텍처로 운영된다. Dockerfile에는 `FROM --platform`을 지정하지 않고, 빌드 시 CLI에서 `--platform linux/amd64`을 전달하는 방식을 사용한다 ([Docker 공식 권장](https://docs.docker.com/reference/build-checks/from-platform-flag-const-disallowed/)).

### 빌드 방법: K8s 원격 빌더 (권장)

K8s management 노드에서 네이티브 amd64로 빌드한다. Apple Silicon 로컬 에뮬레이션 대비 **3~5배 빠르고**, 빌드 결과가 내부망으로 직접 레지스트리에 push된다.

```bash
# 빌더 생성 (최초 1회)
docker buildx create --name k8s-amd64 --driver kubernetes \
    --driver-opt "namespace=rag-bench-test,nodeselector=node-role.kubernetes.io/management=management" \
    --platform linux/amd64
docker buildx inspect k8s-amd64 --bootstrap

# 빌드 + 푸시
docker buildx build --builder k8s-amd64 --platform linux/amd64 --push \
    -t $IMAGE -f k8s/Dockerfile .

# 빌드 완료 후 빌더 제거 (리소스 해제)
docker buildx rm k8s-amd64
```

> `--push` 필수: 원격 빌더에서 `--load`는 지원되지 않음. 이미지가 레지스트리로 직접 push된다.
> 빌더 Pod가 클러스터 리소스를 점유하므로, **빌드 후 반드시 `docker buildx rm`으로 제거**해야 한다.

### 빌드 방법: 로컬 빌드 (대안)

```bash
docker buildx build --platform linux/amd64 -t $IMAGE -f k8s/Dockerfile --load .
docker push $IMAGE
```

로컬 빌드의 장점:
- Dockerfile이 플랫폼에 종속되지 않아 유연성 유지
- BuildKit lint 경고(`FromPlatformFlagConstDisallowed`) 방지
- Apple Silicon(arm64) 로컬 환경에서 크로스 빌드 가능 (단, 에뮬레이션으로 느림)

### 이미지 경량화 전략

멀티스테이지 빌드 + 선택적 의존성으로 이미지 크기를 최소화한다.

| 최적화 | 설명 | 절감 |
|--------|------|------|
| 멀티스테이지 빌드 | 컴파일 도구(`g++`, `git`)를 런타임에서 제거 | 중 |
| PyTorch CPU-only | `--index-url .../whl/cpu`로 CUDA 번들 제거 | ~2.5GB |
| JDK → JRE | `default-jdk-headless` → `default-jre-headless` | ~200MB |
| venv 복사 | `--prefix` 대신 venv 통째 복사로 안정성 확보 | - |
| 불필요 패키지 제외 | 12개 패키지 제거 (`k8s/requirements-worker.txt` 참조) | 중 |

제외된 패키지: `jupyter`, `ipython`, `matplotlib`, `langgraph`, `langchain-ollama`, `langchain-upstage`, `pymupdf4llm`, `python-docx`, `beautifulsoup4`, `lxml`, `rapidfuzz`, `einops`

---

## 6. 리소스 설계

### 6.1 클러스터 노드 구성

| 역할 | 인스턴스 | CPU/MEM | 대수 | Taint | 비고 |
|------|---------|---------|------|-------|------|
| management | m7i/m8i.2xlarge | 8C/32G | 2 | 없음 | **Job 실행 노드** |
| ai-platform | r5a.xlarge | 4C/32G | 2 | `ai-platform: true` | 스케줄링 불가 |
| edge/monitoring | m8i.2xlarge | 8C/32G | 1 | 없음 | 모니터링 워크로드 |

실제 Job이 스케줄링되는 노드는 **management 2대**(taint 없음). 기존 워크로드가 CPU 50~60% 점유 중이므로 리소스 request를 보수적으로 설정해야 한다.

### 6.2 볼륨

| PVC | 용도 | Access Mode | 크기 |
|-----|------|-------------|------|
| `bench-results` | 결과 + 준비 데이터 공유 | ReadWriteMany | 10Gi |
| `model-cache` | HF 모델 공유 캐시 | ReadWriteMany | 50Gi |
| (emptyDir) | Qdrant 인덱스 임시 저장 | Pod-local | 20Gi |

Qdrant 인덱스는 emptyDir에 저장하여:
- PVC I/O 경합 방지
- Pod 종료 시 자동 정리
- 결과 PVC의 용량 절약

### 6.3 리소스 기본값

management 2대의 가용 리소스를 고려한 권장값:

| Phase | CPU req/limit | MEM req/limit | 비고 |
|-------|---------------|---------------|------|
| Prep  | 1/2 | 4Gi/8Gi | HF 다운로드 + Contextual enrichment |
| Bench | 1/2 | 4Gi/8Gi | 임베딩 모델 로딩 + Qdrant 인덱싱 |

> 기본값(2/4 CPU, 8Gi/16Gi)은 management 노드의 기존 워크로드와 경합하여 Pending이 발생한다.
> `--prep-cpu-request 1 --bench-cpu-request 1` 등으로 축소 필요.

### 6.4 시간 예측 (service 프리셋 기준)

| Phase | Job 수 | 예상 소요 | 병렬 시 벽시계 |
|-------|--------|----------|---------------|
| Phase 1 | 4 | 카테고리당 10-30min | 10-30min |
| Phase 2 | 24 | 조합당 5-15min | 5-15min |
| 수집+병합 | 1 | 2-5min | 2-5min |
| **합계** | **29** | | **~17-50min** |

순차 실행 대비: 4카테고리 × 6조합 × 20min ≈ 8시간 → **10배 이상 단축**.

---

## 7. 파일 구조

```
k8s/
├── Dockerfile                          # 멀티스테이지 워커 이미지
├── requirements-worker.txt             # 워커 전용 경량 의존성
├── .dockerignore
├── worker_entrypoint.py                # 워커 (prep/bench 2-Phase)
├── orchestrator.py                     # Job 생성/모니터링/수집/병합
├── ARCHITECTURE.md                     # 이 문서
└── manifests/
    ├── namespace.yaml                  # rag-bench-test 네임스페이스
    ├── results-pvc.yaml                # 결과 공유 PVC
    ├── model-cache-pvc.yaml            # 모델 캐시 공유 PVC
    ├── prep-job-template.yaml          # Phase 1 Job 템플릿
    └── bench-job-template.yaml         # Phase 2 Job 템플릿
```

---

## 8. 사용법

```bash
# .env 로드
eval "$(grep -E '^HARBOR|^OPENAI_API_KEY' .env)"
export IMAGE=$HARBOR_REGISTRY/rag-bench-test/worker:latest

# 이미지 빌드 — K8s 원격 빌더 (권장)
docker buildx create --name k8s-amd64 --driver kubernetes \
    --driver-opt "namespace=rag-bench-test,nodeselector=node-role.kubernetes.io/management=management" \
    --platform linux/amd64
docker buildx inspect k8s-amd64 --bootstrap
docker buildx build --builder k8s-amd64 --platform linux/amd64 --push \
    -t $IMAGE -f k8s/Dockerfile .
docker buildx rm k8s-amd64  # 빌드 후 리소스 해제

# 전체 실행 (4카테고리 × 6조합 = 28 Jobs, 리소스 축소)
python k8s/orchestrator.py --image $IMAGE \
    --prep-cpu-request 1 --prep-cpu-limit 2 \
    --prep-memory-request 4Gi --prep-memory-limit 8Gi \
    --bench-cpu-request 1 --bench-cpu-limit 2 \
    --bench-memory-request 4Gi --bench-memory-limit 8Gi

# dry-run (생성될 Job YAML 확인)
python k8s/orchestrator.py --image $IMAGE --dry-run

# 특정 카테고리만
python k8s/orchestrator.py --image $IMAGE --categories general,legal

# Phase 2만 재실행 (Phase 1 결과 재사용)
python k8s/orchestrator.py --image $IMAGE --skip-prep --run-id <RUN_ID>

# full 프리셋 (5 Dense × 2 Sparse × 2 Reranker = 20 조합 × 4 카테고리 = 80 Jobs)
python k8s/orchestrator.py --image $IMAGE --preset full
```

### 클러스터 사전 요구사항

1. **ReadWriteMany PVC** 지원 StorageClass (`efs-zcp`)
2. `.env` 파일: `OPENAI_API_KEY`, `HARBOR_REGISTRY`, `HARBOR_USER`, `HARBOR_CLI_SECRET`
3. `kubectl` + `docker buildx` 로컬 설치, 클러스터 접근 설정 (MFA 인증)

---

## 9. 검토된 안전성 이슈 및 대응

| 이슈 | 심각도 | 대응 |
|------|--------|------|
| 오케스트레이터/워커 경로 불일치 | CRITICAL | `COMBO_LABEL` 환경변수를 단일 진실 공급원으로 통일 |
| PVC 쓰기 가시성 지연 | CRITICAL | Phase 1→2 사이 검증 Pod로 데이터 확인 |
| 부분 실패 시 결과 누락 무감지 | CRITICAL | 기대/실제 조합 수 비교 + WARN 로깅 |
| 동시 PVC 쓰기 시 파일 손상 | HIGH | 원자적 쓰기 (tmp → rename) + DONE 시그널 |
| 프로세스 격리 시 싱글톤 상태 | LOW | 각 Pod 독립 프로세스 — 문제 없음 확인 |
| parent_pairs Phase 2 불필요 | LOW | 제공하되 미사용 — 무해. 추후 최적화 가능 |
| BM25 vocab 불일치 | MEDIUM | Qdrant 경로가 조합별 고유 — 충돌 없음 확인 |
| ColBERT 스레드 안전성 | LOW | Pod당 단일 프로세스 — Lock 불필요하나 무해 |
| Contextual 캐시 경로 오류 | HIGH | 패키지 내부 `_benchdata/` 미존재 → `cache_dir`을 PVC 경로로 명시 지정하여 해결 |
