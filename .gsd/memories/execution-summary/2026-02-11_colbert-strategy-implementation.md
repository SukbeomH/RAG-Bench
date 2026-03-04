---
title: "ColBERT Late Interaction 전략 구현 (PyLate 기반)"
tags:
  - execution
  - summary
  - colbert
  - pylate
  - retrieval
  - strategy
type: execution-summary
created: 2026-02-11T14:30:00+09:00
contextual_description: "PyLate 백엔드로 ColBERTStrategy 전체 구현 완료. brute-force MaxSim + Voyager ANN 지원, LangChain Retriever 호환."
keywords:
  - ColBERT
  - PyLate
  - jina-colbert-v2
  - MaxSim
  - brute-force
  - Voyager
  - Late Interaction
  - BaseRetriever
  - rank.rerank
related:
  - 2026-02-11_project-memory-sync
---

## ColBERT Late Interaction 전략 구현 (PyLate 기반)

### 실행 내용
스텁 상태였던 `rag_bench/strategies/colbert.py`를 PyLate 백엔드로 전체 구현.

### 변경 파일
| 파일 | 변경 |
|------|------|
| `rag_bench/strategies/colbert.py` | 스텁 → 전체 구현 (231줄) |
| `pyproject.toml` | `pylate>=1.0`, `einops>=0.8.2` 의존성 추가 |

### 구현 상세
- **ColBERTRetriever** (~10 LOC): `BaseRetriever` 상속, `strategy.retrieve()` 위임, `ConfigDict(arbitrary_types_allowed=True)`
- **ColBERTStrategy** (~170 LOC):
  - 기본 모델: `jinaai/jina-colbert-v2` (89개 언어, 한국어 포함)
  - Lazy 모델 로드 (`_ensure_initialized`), CUDA/MPS/CPU 자동 감지
  - **Brute-force 모드 (기본)**: `pylate.rank.rerank()`으로 MaxSim 스코어링
  - **Voyager 인덱스 모드** (`use_index=True`): `pylate.indexes.Voyager` ANN 인덱스
  - 메타데이터 완전 보존 (`.metadata.copy()`), k clamp 처리
  - `trust_remote_code=True` 필수 (jina-colbert-v2 커스텀 코드)

### 해결된 이슈
1. **trust_remote_code**: jina-colbert-v2가 커스텀 XLM-RoBERTa 사용 → `models.ColBERT(trust_remote_code=True)`
2. **einops 누락**: 모델 커스텀 코드가 einops 요구 → `pyproject.toml`에 추가
3. **HuggingFace XET CDN 오류**: `CAS service error: Request failed after 5 retries` → `HF_HUB_DISABLE_XET=1` 환경변수로 해결

### 검증 결과
- jina-colbert-v2: 한국어 쿼리 검색, 메타데이터 보존, LangChain Retriever 호환 모두 통과
- all-MiniLM-L6-v2: 경량 모델로 전체 로직 검증 완료

### 커밋
```
73dd2b9 feat: ColBERT Late Interaction 검색 전략 구현 (PyLate 기반)
```

### 현재 전략 구현 상태
| 전략 | 상태 |
|------|------|
| DenseSparseStrategy | 완료 (6가지 조합) |
| ColBERTStrategy | 완료 (brute-force + Voyager) |
| GraphRAGStrategy | 스텁 |
