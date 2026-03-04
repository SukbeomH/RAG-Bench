---
title: "mypy implicit Optional 패턴 — str = None 금지, Optional[str] = None 사용"
tags:
  - pattern
  - learning
  - mypy
  - python
  - type-annotation
type: pattern-discovery
created: "2026-02-19T00:00:00+09:00"
contextual_description: "mypy strict에서 str=None 파라미터는 에러. Optional[str]=None으로 명시해야 하며, monkey-patch는 type: ignore[method-assign] 처리"
keywords:
  - mypy
  - implicit Optional
  - Optional[str]
  - method-assign
  - type: ignore
  - monkey-patch
  - run_tracker
  - colab_config
related:
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
---

## mypy implicit Optional 패턴

### 패턴 1: implicit Optional 금지 (PEP 484)
```python
# ❌ 에러: Incompatible default for argument (str = None)
def start_build(self, label: str, dense: str = None, reranker: str = None):
    ...

# ✅ 수정: Optional[str] = None
def start_build(self, label: str, dense: Optional[str] = None, reranker: Optional[str] = None):
    ...
```
mypy는 기본값으로 `no_implicit_optional=True` 적용 → `str = None`이면 에러.

### 패턴 2: monkey-patch는 type: ignore[method-assign]
```python
# 런타임 동적 메서드 교체 (Colab 환경 패치)
DenseSparseStrategy._init_dense = _patched_init_dense  # type: ignore[method-assign]
DenseSparseStrategy._init_qdrant = _patched_init_qdrant  # type: ignore[method-assign]
```
mypy는 클래스 메서드를 직접 재할당하는 것을 허용하지 않음.

### 패턴 3: [None] * n 리스트 타입
```python
# ❌ List[None]으로 추론됨
results = [None] * len(self.queries)
results[idx] = future.result()  # 에러: dict 할당 불가

# ✅ Optional 명시
results: List[Optional[dict]] = [None] * len(self.queries)
results[idx] = future.result()  # OK
return results  # type: ignore[return-value]  # 반환 타입이 List[dict]인 경우
```

### 패턴 4: Optional 클래스 속성 재정의 시 첫 할당에 어노테이션
```python
# ❌ 두 번째 정의에서 no-redef + assignment 에러
if condition:
    self._combo_id = combo_id           # int 추론
    self._combo = combo
else:
    self._combo_id: Optional[int] = None  # no-redef 에러
    self._combo: Optional[dict] = None

# ✅ 첫 할당에 Optional 어노테이션
if condition:
    self._combo_id: Optional[int] = combo_id  # 첫 번째 정의에서 Optional
    self._combo: Optional[dict] = combo
else:
    self._combo_id = None  # OK (이미 Optional[int] 타입)
    self._combo = None
```

### 패턴 5: TYPE_CHECKING guard로 전방 참조
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_bench.evaluation.evaluator import ExtendedRAGEvaluator

# 문자열 전방 참조로 사용
evaluator: Optional["ExtendedRAGEvaluator"] = None
```
런타임 import 없이 타입 힌트만 제공.

### 적용 파일
- `rag_bench/run_tracker.py`: implicit Optional 수정
- `rag_bench/strategies/dense_sparse.py`: 클래스 속성 첫 할당 어노테이션
- `rag_bench/runner.py`: TYPE_CHECKING guard, List[Optional] 패턴
- `rag_bench_colab/colab_config.py`: method-assign type: ignore
