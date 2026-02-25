# Session Handoff Document

> 작성일: 2026-02-25 | 브랜치: `master` | 세션: K8s 병렬 벤치마크 구축 + 모델 서빙 설계

---

## 1. 프로젝트 개요

**RAG Bench** — 문서 유형별 최적 Dense×Sparse 임베딩 조합을 찾기 위한 서비스 벤치마크 시스템.

- 5개 문서 카테고리: GENERAL / LEGAL / BUSINESS / MEDICAL / TECHNICAL
- 6개 조합: 3 Dense(kosimcse, e5, bge-m3) × 2 Sparse(korean_bm25, splade)
- 고정 파이프라인: ColBERT Reranker + Contextual Retrieval
- 평가: RAGAS core_only (Recall 0.35, Precision 0.30, Faithfulness 0.20, Relevancy 0.15)

---

## 2. 이번 세션 작업 요약

### K8s 병렬 벤치마크 시스템 구축

2-Phase 아키텍처로 K8s에서 벤치마크를 병렬 실행하는 시스템을 구축함.

#### 생성/수정된 파일

| 파일 | 설명 |
|------|------|
| `k8s/Dockerfile` | 멀티스테이지 빌드, CPU-only torch, venv 복사 방식 |
| `k8s/requirements-worker.txt` | 워커 전용 경량 의존성 (12개 패키지 제외) |
| `k8s/worker_entrypoint.py` | 2-Phase 워커 (prep/bench) |
| `k8s/orchestrator.py` | Job 생성/모니터링/수집/병합 오케스트레이터 |
| `k8s/ARCHITECTURE.md` | 설계 문서 (9개 섹션) |
| `k8s/DEPLOY_GUIDE.md` | 단계별 배포 가이드 |
| `k8s/manifests/namespace.yaml` | `rag-bench-test` 네임스페이스 |
| `k8s/manifests/results-pvc.yaml` | 결과 공유 PVC (EFS, 10Gi RWX) |
| `k8s/manifests/model-cache-pvc.yaml` | 모델 캐시 PVC (EFS, 50Gi RWX) |
| `k8s/manifests/prep-job-template.yaml` | Phase 1 Job 템플릿 |
| `k8s/manifests/bench-job-template.yaml` | Phase 2 Job 템플릿 |
| `.dockerignore` | Docker 빌드 컨텍스트 제외 목록 |

#### 클러스터 상태 (실배포 완료)

| 리소스 | 상태 |
|--------|------|
| 네임스페이스 `rag-bench-test` | 생성 완료 |
| PVC `bench-results` (10Gi) | Bound |
| PVC `model-cache` (50Gi) | Bound |
| Secret `bench-secrets` (OPENAI_API_KEY) | 생성 완료 |
| Secret `harbor-cred` (ImagePullSecret) | 생성 완료 |
| 이미지 `worker:latest` | Harbor 푸시 완료 |

#### 발견 및 수정한 이슈

| 이슈 | 원인 | 수정 |
|------|------|------|
| `FromPlatformFlagConstDisallowed` | Dockerfile `FROM --platform=linux/amd64` 상수 | `FROM`에서 제거, CLI `--platform`만 사용 |
| Contextual 캐시 경로 오류 | `_benchdata/` 디렉토리 미존재 (dockerignore 제외) | `cache_dir`을 PVC 경로로 명시 지정 |
| Pod Pending (Insufficient cpu) | 기본 리소스(2/4 CPU)가 management 노드 가용량 초과 | CPU request 1, limit 2로 축소 |
| K8s 빌더 리소스 충돌 | buildx Pod가 클러스터 CPU 점유 | 빌드 후 `docker buildx rm` 필수 |
| pip torch 덮어쓰기 | requirements의 의존성이 GPU torch 재설치 | 기능 문제 없음 (CPU fallback), 추후 최적화 |

### 모델 서빙 설계 (논의 단계)

임베딩/리랭커 모델을 네임스페이스 내 상주 Pod로 서빙하고 Job에서 API 호출하는 구조 논의.

- **TEI (Text Embeddings Inference)**: HF 공식, OpenAI 호환 API, CPU 모드 지원 → 추천
- **FastAPI 커스텀**: sentence-transformers 직접 서빙, 완전 제어 가능
- GPU 노드 없음 → CPU 서빙만 가능
- 장점: 모델 로딩 1회, Job 리소스 대폭 축소, 동시 실행 수 증가

---

## 3. 현재 진행 상태

### 테스트 실행 중

```
오케스트레이터 실행 중:
  카테고리: general (소규모 테스트)
  max_corpus: 100, max_queries: 10
  리소스: CPU 1/2, MEM 4Gi/8Gi
```

### 남은 작업

#### P0: 즉시 필요
1. **테스트 실행 결과 확인** — general 카테고리 소규모 테스트 성공 여부
2. **전체 카테고리 실행** — 4카테고리 × 6조합 = 28 Jobs
3. ~~**torch CPU-only 이미지 최적화**~~ — **완료**: Dockerfile에 CPU torch 강제 재설치 + nvidia 패키지 제거 추가

#### P1: 모델 서빙 구현
4. ~~**TEI 기반 임베딩 서빙 Pod 구현**~~ — **완료**: `k8s/manifests/tei-deployment.yaml` 생성
5. ~~**worker_entrypoint.py에 API 호출 모드 추가**~~ — **완료**: `EMBEDDING_API_URL` 환경변수 + dense_sparse.py TEI 분기
6. **벤치마크 결과 수집 + 병합 리포트 확인**

#### P2: 개선
7. ~~Dockerfile torch CPU-only 강제~~ — **완료** (P0-3에서 처리)
8. ~~오케스트레이터 stdout 버퍼링 문제 해결~~ — **완료**: `python -u` 플래그 추가
9. 로컬 빌드 비활성화된 상태 확인 (b9de338 빌드는 중단 필요할 수 있음)
10. **TEI 실제 배포 테스트** — `--tei --dry-run`으로 YAML 확인 후 실배포

---

## 4. 클러스터 정보

### 노드 구성

| 역할 | 인스턴스 | CPU/MEM | 대수 | Taint |
|------|---------|---------|------|-------|
| management | m7i/m8i.2xlarge | 8C/32G | 2 | 없음 (Job 실행) |
| ai-platform | r5a.xlarge | 4C/32G | 2 | `ai-platform: true` |
| edge/monitoring | m8i.2xlarge | 8C/32G | 1 | 없음 |

- EKS 클러스터: `zcp-ags-cp-eks` (ap-northeast-2)
- MFA 인증 필요 (AWS MFA, 토큰 주기적 만료)
- management 노드 기존 워크로드 CPU 50~60% 점유

### 환경변수 (.env)

| 변수 | 용도 |
|------|------|
| `HARBOR_REGISTRY` | Harbor 레지스트리 URL |
| `HARBOR_USER` | Harbor 사용자 (`cloudzcp-admin`) |
| `HARBOR_CLI_SECRET` | Harbor 비밀번호 |
| `OPENAI_API_KEY` | OpenAI API 키 |

---

## 5. 핵심 파일 맵

```
k8s/
├── Dockerfile                    # 멀티스테이지 워커 이미지 (syntax=docker/dockerfile:1)
├── requirements-worker.txt       # 워커 전용 의존성 (torch는 Dockerfile에서 CPU-only 설치)
├── worker_entrypoint.py          # prep: HF→chunk→enrich→serialize / bench: deserialize→index→pass1+2
├── orchestrator.py               # Job CRUD, 모니터링, 결과 수집/병합
├── ARCHITECTURE.md               # 설계 문서
├── DEPLOY_GUIDE.md               # 배포 가이드
└── manifests/
    ├── namespace.yaml
    ├── results-pvc.yaml          # EFS 10Gi RWX
    ├── model-cache-pvc.yaml      # EFS 50Gi RWX
    ├── prep-job-template.yaml    # Phase 1 템플릿
    └── bench-job-template.yaml   # Phase 2 템플릿

rag_bench/
├── datasets/hf_loader.py        # HF 데이터셋 로더
├── strategies/
│   ├── contextual_retrieval.py   # cache_dir 파라미터로 캐시 경로 지정
│   ├── dense_sparse.py           # Dense+Sparse 전략
│   ├── colbert_rerank.py         # ColBERT 리랭커
│   └── flashrank_rerank.py       # FlashRank 리랭커
├── combo/                        # ComboSpec, builder, cache
├── runner.py                     # BenchmarkRunner
├── evaluation/                   # RAGAS 평가
└── config.py                     # BENCH_DATA_DIR, make_llm 등
```

---

## 6. 주의 사항

### K8s 빌드
- **K8s 원격 빌더 사용 후 반드시 `docker buildx rm k8s-amd64`** (리소스 해제)
- 원격 빌더는 `--push` 필수 (`--load` 미지원)
- 로컬 빌드 시 에뮬레이션으로 30분+ 소요

### 리소스
- CPU request는 **1** 이하로 설정해야 스케줄링 가능
- 기본값(2/4 CPU)은 management 노드 경합으로 Pending 발생
- ai-platform 노드는 taint로 스케줄링 불가

### MFA
- kubectl 명령 실행 전 MFA 토큰 갱신 필요
- 비대화형 셸에서 MFA 프롬프트 차단됨 → 사전에 인증 필요

### 코드
- `contextual_retrieval.py`의 기본 cache_dir은 `_benchdata/` (패키지 내부)
- K8s worker에서는 PVC 경로로 명시 지정 필수: `cache_dir=str(results_dir / category / "ctx_cache")`
- orchestrator.py stdout이 버퍼링됨 → 백그라운드 실행 시 실시간 출력 안 됨

---

## 7. 참조 문서

| 문서 | 경로/URL |
|------|----------|
| 설계 문서 | `k8s/ARCHITECTURE.md` |
| 배포 가이드 | `k8s/DEPLOY_GUIDE.md` |
| 프로젝트 메모리 | `MEMORY.md` |
| Docker 플랫폼 가이드 | https://docs.docker.com/reference/build-checks/from-platform-flag-const-disallowed/ |
| TEI (임베딩 서버) | https://github.com/huggingface/text-embeddings-inference |
