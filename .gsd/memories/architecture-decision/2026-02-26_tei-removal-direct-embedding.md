---
title: "TEI 서빙 제거 — 직접 임베딩 방식 확정"
tags:
  - arch
  - decision
  - k8s
  - embedding
  - tei
type: architecture-decision
created: 2026-02-26T09:50:00+09:00
contextual_description: "TEI(Text Embeddings Inference) 서빙이 현 클러스터에서 OOM/SIGSEGV로 사용 불가하여 코드 전면 제거, 워커 Pod 내 직접 HF 임베딩 방식으로 확정"
keywords:
  - TEI
  - embedding
  - OOMKilled
  - SIGSEGV
  - AMD EPYC
  - Intel MKL
  - direct embedding
related:
  - 2026-02-26_tei-oom-sigsegv-cluster-incompatibility
---

## TEI 서빙 제거 — 직접 임베딩 방식 확정

### 결정
TEI(Text Embeddings Inference) 서빙 코드를 전면 제거하고, 워커 Pod 내에서 HuggingFace 모델을 직접 로딩하는 방식으로 확정.

### 근거
1. **bge-m3 OOMKilled**: float32 ONNX warm-up 시 12Gi 메모리도 부족 (568M param)
2. **kosimcse SIGSEGV**: Intel MKL SGEMM과 AMD EPYC(r5a.xlarge) CPU 비호환 (exit 139)
3. GPU 노드 없이 TEI CPU 모드는 현 클러스터에서 사용 불가

### 대안 검토
| 방안 | 평가 |
|------|------|
| A: 직접 임베딩 (채택) | 즉시 실행 가능, 추가 인프라 불필요 |
| B: TEI + Intel 노드 전용 | Intel 노드 여유 부족 (48Gi 필요) |
| C: TEI + GPU 노드 추가 | 추가 비용, 프로비저닝 시간 |

### 영향 범위
- `k8s/orchestrator.py`: --tei 플래그, TEI deploy/cleanup 함수 제거
- `rag_bench/strategies/dense_sparse.py`: embedding_api_url 파라미터 제거
- `rag_bench/combo/cache.py`: CacheConfig.embedding_api_url 필드 제거
- `k8s/worker_entrypoint.py`: EMBEDDING_API_URL 처리 제거
- `k8s/manifests/tei-deployment.yaml`: 파일 삭제
- `k8s/manifests/bench-job-template.yaml`: EMBEDDING_API_URL 환경변수 제거
