# TEI 서빙 구현 + Dockerfile CPU-only 최적화

- **date**: 2026-02-25
- **branch**: master
- **status**: completed

## 작업 내용

### 1. Dockerfile torch CPU-only 최적화 (P0-3 + P2-7)
- `k8s/Dockerfile`: requirements 설치 후 CPU-only torch `--force-reinstall --no-deps` + nvidia 패키지 제거
- 멀티스테이지 빌드이므로 최종 venv 상태만 런타임에 복사 → 이미지 ~2.5GB 절감

### 2. stdout 버퍼링 해결 (P2-8)
- `k8s/Dockerfile`: ENTRYPOINT에 `python -u` 플래그 추가

### 3. TEI 서빙 Pod manifest (P1-4)
- `k8s/manifests/tei-deployment.yaml`: Deployment + Service 템플릿 (모델별 치환)
- TEI 이미지: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7`
- 3개 모델: kosimcse(8081), e5(8082), bge-m3(8083)

### 4. Worker API 호출 모드 (P1-5)
- `rag_bench/strategies/dense_sparse.py`: `embedding_api_url` 파라미터, `_init_dense()` TEI 분기
- `rag_bench/combo/cache.py`: `CacheConfig.embedding_api_url` 필드
- `k8s/worker_entrypoint.py`: `EMBEDDING_API_URL` 환경변수
- `k8s/manifests/bench-job-template.yaml`: `${EMBEDDING_API_URL}` 플레이스홀더
- `k8s/orchestrator.py`: `--tei` CLI 플래그, `deploy_tei_services()`, `cleanup_tei_services()`, `get_tei_url()`

## 변경 파일
| 파일 | 변경 유형 |
|------|-----------|
| `k8s/Dockerfile` | 수정 (CPU torch 강제 + -u 플래그) |
| `k8s/manifests/tei-deployment.yaml` | 신규 (TEI Deployment + Service) |
| `k8s/manifests/bench-job-template.yaml` | 수정 (EMBEDDING_API_URL 추가) |
| `k8s/orchestrator.py` | 수정 (TEI 배포/정리/URL 주입) |
| `k8s/worker_entrypoint.py` | 수정 (EMBEDDING_API_URL 전달) |
| `rag_bench/strategies/dense_sparse.py` | 수정 (TEI 분기) |
| `rag_bench/combo/cache.py` | 수정 (embedding_api_url 필드) |
| `SESSION_HANDOFF.md` | 수정 (진행 상태 갱신) |

## 남은 작업
- P0-1: 테스트 실행 결과 확인 (클러스터 접근 필요)
- P0-2: 전체 카테고리 실행
- P1-6: 벤치마크 결과 수집 + 병합 리포트 확인
- P2-10: TEI 실제 배포 테스트 (`--tei --dry-run`)
