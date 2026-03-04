---
title: "Pass 2 ComboSpec 매칭 실패 — _strategy_name_from_spec dense 키 미해석"
tags:
  - debug
  - root-cause
  - colab
  - rag-bench
  - pass2
type: root-cause
created: "2026-02-19T00:00:00+09:00"
contextual_description: "_strategy_name_from_spec이 'minilm' 키를 그대로 사용해 실제 전략명 'DS(all-MiniLM-L6-v2+...)' 와 불일치 발생"
keywords:
  - _strategy_name_from_spec
  - DENSE_MODELS
  - ComboSpec
  - Pass2
  - strategy matching
  - ColabBenchmarkRunner
  - minilm
  - all-MiniLM-L6-v2
related:
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
---

## Pass 2 ComboSpec 매칭 실패 — `_strategy_name_from_spec` dense 키 미해석

### 증상
Pass 2 RAGAS 평가 시 아래 로그 출력:
```
[Skip] DS(all-MiniLM-L6-v2+fastembed_bm25): ComboSpec 매칭 실패
[Skip] FlashRank Rerank (DS(all-MiniLM-L6-v2+fastembed_bm25)): ComboSpec 매칭 실패
[Skip] FlashRank Rerank (DS(bge-m3+fastembed_bm25)): ComboSpec 매칭 실패
```
bge-m3 기본 전략만 매칭 성공, 나머지는 모두 스킵.

### 근본 원인
`colab_runner.py`의 `_strategy_name_from_spec()`:

```python
# 버그 코드
def _strategy_name_from_spec(self, spec) -> str:
    dense_short = spec.dense      # "minilm" (ComboSpec 키)
    if "/" in dense_short:
        dense_short = dense_short.split("/")[-1]
    return f"DS({dense_short}+{spec.sparse})"
    # → "DS(minilm+fastembed_bm25)"  ← 실제 전략명과 불일치!
```

실제 `DenseSparseStrategy.name` 프로퍼티:
```python
dense_short = self._dense_model.split("/")[-1]
# _dense_model = DENSE_MODELS.get("minilm") = "sentence-transformers/all-MiniLM-L6-v2"
# → dense_short = "all-MiniLM-L6-v2"
# → name = "DS(all-MiniLM-L6-v2+fastembed_bm25)"
```

**추가 문제**: 리랭커 래퍼(FlashRank, ColBERT) prefix도 미처리.

### 수정 코드
```python
def _strategy_name_from_spec(self, spec) -> str:
    from rag_bench.strategies.dense_sparse import DENSE_MODELS
    dense_model = DENSE_MODELS.get(spec.dense, spec.dense)  # 키 → 실제 모델명
    dense_short = dense_model.split("/")[-1]
    base_name = f"DS({dense_short}+{spec.sparse})"
    if spec.reranker == "flashrank":
        reranked_name = f"FlashRank Rerank ({base_name})"
    elif spec.reranker == "colbert":
        reranked_name = f"ColBERT Rerank ({base_name})"
    else:
        reranked_name = base_name
    if spec.llm_support == "contextual":
        return f"Contextual Retrieval ({reranked_name})"
    return reranked_name
```

### 핵심 교훈
- `ComboSpec.dense` = 짧은 키 ("minilm"), `DenseSparseStrategy._dense_model` = 실제 HF 경로
- `DENSE_MODELS` 딕셔너리: `{"minilm": "sentence-transformers/all-MiniLM-L6-v2", ...}`
- 전략 이름은 항상 **실제 HF 모델의 마지막 경로 세그먼트** 기반

### 파일
- `rag_bench_colab/colab_runner.py`: `_strategy_name_from_spec()` 수정
- `rag_bench/strategies/dense_sparse.py`: `DENSE_MODELS` 딕셔너리 참조
