---
title: "Upstage 임베딩 모델명 변경 — solar-embedding-1-query → embedding-query"
tags:
  - pattern-discovery
  - upstage
  - embedding
  - model-name
  - breaking-change
type: pattern-discovery
created: 2026-02-23T12:00:00+09:00
contextual_description: "Upstage API가 solar-embedding-1-query 모델명을 더 이상 지원하지 않음. 새 alias인 embedding-query / embedding-passage 사용 필요. dense_sparse.py, upstage_embed.py 두 파일 수정."
keywords:
  - upstage
  - solar-embedding-1-query
  - embedding-query
  - embedding-passage
  - BadRequestError
  - invalid_request_body
  - dense_sparse.py
  - upstage_embed.py
related:
  - 2026-02-23_full-bench-run-upstage-retry-setup
---

## Upstage 임베딩 모델명 변경 (2026-02-23)

### 증상
```
openai.BadRequestError: Error code: 400 - {'error': {'message':
  'The requested model is invalid or no longer supported.
   You can find the list of available models on our models page
   at https://console.upstage.ai/docs/models.'}}
```
upstage 12개 조합 전체 실패.

### 원인
Upstage가 버전별 모델명을 폐기하고 alias 체계로 변경:

| 구 모델명 | 신 alias |
|-----------|---------|
| `solar-embedding-1-query` | `embedding-query` |
| `solar-embedding-1-large-query` | `embedding-query` |
| `solar-embedding-1-large-passage` | `embedding-passage` |

### 수정 파일 및 내용

**`rag_bench/strategies/dense_sparse.py`**
```python
# DENSE_MODEL_IDS
"upstage": "embedding-query"   # 구: "solar-embedding-1-query"

# DENSE_DIMS
"embedding-query": 4096        # 구: "solar-embedding-1-query": 4096

# _init_qdrant()
elif "embedding-query" in model_spec or "embedding-passage" in model_spec or "solar-embedding" in model_spec:
    self._dense_embeddings = UpstageEmbeddings(model=model_spec)  # 구: 하드코딩
```

**`rag_bench/strategies/upstage_embed.py`**
```python
model: str = "embedding-passage"       # 구: "solar-embedding-1-large-passage"
query_model: str = "embedding-query"   # 구: "solar-embedding-1-large-query"
```

### 참고
- Upstage 공식 문서: https://console.upstage.ai/docs/capabilities/embeddings
- alias는 최신 버전을 자동으로 가리키므로 하드코딩보다 안전
