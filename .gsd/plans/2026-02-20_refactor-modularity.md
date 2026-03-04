# Refactor: 코드 구조 / 모듈화 / 재사용성 개선 계획

**작성일**: 2026-02-20
**근거**: 아키텍처 리뷰 (2026-02-20) — CRITICAL 2건, MAJOR 6건
**목표**: 레이어 분리, 중복 제거, 캡슐화 강화로 유지보수성 및 재사용성 확보

---

## 골-역방향 검증 (Goal-Backward Verification)

**최종 목표**: `rag_bench/` 코어 패키지는 스크립트/Colab에 독립적이고,
새 전략·모델·환경 추가 시 수정 범위가 명확하게 한정된다.

- `rag_bench/` 임포트만으로 전략 빌드/실행/평가가 완결 → Phase 1, 2
- 5개 Retriever 래퍼 → 1개로 → Phase 3
- 디바이스 감지 5곳 → 1곳 → Phase 4
- Colab이 monkey-patch 없이 config 주입 → Phase 2
- 중복 유틸리티 (`_load_qa_dataset`, 리포트, 답변 생성) → 공유 모듈 → Phase 5
- LLM 모델명 하드코딩 4곳 → `config.py` 상수 1곳 → Phase 6

---

## Phase 1 — `rag_bench/combo/` 패키지 추출 (CRITICAL: God Script)

**문제**: `run_all_combos.py`(1,348줄)에 핵심 비즈니스 로직이 스크립트와 혼재.
`IndexCacheManager`, `ComboSpec` 등이 `colab_runner.py`에서도 직접 import됨.

### Tasks

#### 1-1. `rag_bench/combo/__init__.py` 생성
- 빈 `__init__.py` + 공개 API 정의

#### 1-2. `rag_bench/combo/spec.py` — ComboSpec + 조합 생성
**이동 대상** (`run_all_combos.py` → `combo/spec.py`):
- `ComboSpec` dataclass (라인 59-93)
- `PRESETS` dict (라인 99-118)
- `generate_valid_combinations()` (라인 121-133)

```
rag_bench/combo/spec.py
  class ComboSpec
  PRESETS
  generate_valid_combinations()
```

#### 1-3. `rag_bench/combo/cache.py` — CacheConfig + IndexCacheManager
**이동 대상** (`run_all_combos.py` → `combo/cache.py`):
- `CacheConfig` dataclass (라인 141-151)
- `IndexCacheManager` dataclass (라인 158-249)

```
rag_bench/combo/cache.py
  class CacheConfig
  class IndexCacheManager
```

#### 1-4. `rag_bench/combo/builder.py` — 전략 팩토리
**이동 대상** (`run_all_combos.py` → `combo/builder.py`):
- `build_strategy_from_spec()` (라인 257-297)
- `_try_build_dense_sparse()` (라인 305-344)
- `_try_build_contextual()` (라인 348-365)
- `_try_build_reranker()` (라인 368-397)
- `_safe_build()` (라인 401-445)

```
rag_bench/combo/builder.py
  build_strategy_from_spec()
  (내부 헬퍼들은 모듈-프라이빗 _함수 유지)
```

#### 1-5. `run_all_combos.py` import 경로 업데이트
```python
# 변경 전
from rag_bench.scripts.run_all_combos import ComboSpec, IndexCacheManager
# 변경 후
from rag_bench.combo.spec import ComboSpec
from rag_bench.combo.cache import IndexCacheManager
from rag_bench.combo.builder import build_strategy_from_spec
```

#### 1-6. `colab_runner.py` import 경로 업데이트
- 동일하게 `rag_bench.combo.*` 경로로 업데이트

#### 1-7. `rag_bench/__init__.py` 공개 API 노출
```python
from rag_bench.combo.spec import ComboSpec, PRESETS, generate_valid_combinations
from rag_bench.combo.cache import CacheConfig, IndexCacheManager
from rag_bench.combo.builder import build_strategy_from_spec
```

**검증 기준**:
- `python -c "from rag_bench.combo.spec import ComboSpec"` 성공
- `python -c "from rag_bench.combo.cache import IndexCacheManager"` 성공
- `python -c "from rag_bench.combo.builder import build_strategy_from_spec"` 성공
- `colab_runner.py` 실행 시 ImportError 없음

---

## Phase 2 — Monkey-Patch → 생성자 파라미터 + Config DI (CRITICAL)

**문제**: `colab_config.py`가 `DenseSparseStrategy._init_dense` 메서드 자체를 교체하여
원본 코드 변경 시 무조건 깨지고 추적 불가.

### Tasks

#### 2-1. `DenseSparseStrategy.__init__()` 에 `device` 파라미터 추가
**대상 파일**: `rag_bench/strategies/dense_sparse.py`

```python
# 변경 전
def __init__(self, dense_model, sparse_type, qdrant_path, ...):
    ...
    self._dense_device = "cpu"   # 하드코딩

# 변경 후
def __init__(self, dense_model, sparse_type, qdrant_path, ..., device: Optional[str] = None):
    ...
    self._dense_device = device or _detect_local_device()
```

#### 2-2. `rag_bench/utils/device.py` 신설 (Phase 4와 연계)
```python
# rag_bench/utils/device.py
def detect_device() -> str:
    """CUDA > MPS > CPU 순으로 자동 감지."""
    ...
```

#### 2-3. `SpladeEncoder.__init__()` 의 `device` 파라미터 활용 확인
- 이미 `device` 파라미터가 있으므로 `detect_device()` 기본값 설정으로 통일

#### 2-4. `IndexCacheManager.config.colbert_device` → `detect_device()` 기본값
**대상**: `combo/cache.py` `CacheConfig.colbert_device` 기본값

#### 2-5. `colab_config.py`의 `patch_dense_device()` 제거
**대상**: `colab_config.py` 라인 284-310

```python
# 제거
DenseSparseStrategy._init_dense = _patched_init_dense

# 대체: colab_runner.py에서 strategy 생성 시 device 명시
strategy = DenseSparseStrategy(..., device=get_device())
```

#### 2-6. `colab_config.py`의 `patch_rag_bench_config()` → 환경 변수 기반으로 전환
**대상**: `colab_config.py` 라인 150-197

```python
# 대체 방향: config.py가 환경 변수를 우선 참조
# config.py
BENCH_DATA_DIR = Path(os.environ.get("RAG_BENCH_DATA_DIR", str(PROJECT_ROOT / "_benchdata")))
```

`colab_config.py`는 환경 변수를 설정하는 역할로만 축소:
```python
os.environ["RAG_BENCH_DATA_DIR"] = str(DRIVE_BENCHDATA_DIR)
```

#### 2-7. `patch_qdrant_memory_mode()` 제거
`DenseSparseStrategy.__init__()`에 `qdrant_mode: str = "disk"` 파라미터 추가,
Colab에서는 `qdrant_mode="memory"` 또는 `"ephemeral"` 명시 전달.

**검증 기준**:
- `colab_config.py`에 `monkey-patch`, `__func__`, `_patched` 패턴이 없음
- `DenseSparseStrategy(device="cuda")` 생성 시 CUDA 사용 확인
- Colab 환경에서 monkey-patch 없이 동일 동작 확인

---

## Phase 3 — BaseRetriever 래퍼 5중 복사 → StrategyRetriever 통합 (MAJOR)

**문제**: `ColBERTRetriever`, `ColBERTRerankRetriever`, `FlashRankRerankRetriever`,
`ContextualRetrievalRetriever`, `_UpstageRetriever` 5개가 동일 패턴을 각자 정의.

### Tasks

#### 3-1. `base.py`에 `StrategyRetriever` 추가
```python
# rag_bench/base.py 추가
from pydantic import ConfigDict
class StrategyRetriever(BaseRetriever):
    """BaseRAGStrategy를 LangChain Retriever로 래핑하는 제네릭 클래스."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strategy: "BaseRAGStrategy"
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.strategy.retrieve(query, k=self.k)
```

#### 3-2. 각 전략의 `get_retriever()` 구현 교체
**대상 파일 5개**:
- `strategies/colbert.py` — `ColBERTRetriever` 클래스 제거
- `strategies/colbert_rerank.py` — `ColBERTRerankRetriever` 클래스 제거
- `strategies/flashrank_rerank.py` — `FlashRankRerankRetriever` 클래스 제거
- `strategies/contextual_retrieval.py` — `ContextualRetrievalRetriever` 클래스 제거
- `strategies/upstage_embed.py` — `_UpstageRetriever` 클래스 제거

각 전략의 `get_retriever()`:
```python
# 변경 전
def get_retriever(self, k: int = 5) -> BaseRetriever:
    return ColBERTRetriever(strategy=self, k=k)

# 변경 후
from rag_bench.base import StrategyRetriever
def get_retriever(self, k: int = 5) -> BaseRetriever:
    return StrategyRetriever(strategy=self, k=k)
```

#### 3-3. `dense_sparse.py`의 `get_retriever()`도 동일 패턴 통일 확인

**검증 기준**:
- `grep -r "class.*Retriever.*BaseRetriever" rag_bench/strategies/` → 결과 없음
- 각 전략의 `get_retriever().get_relevant_documents("test")` 정상 동작

---

## Phase 4 — 디바이스 감지 일원화 (MAJOR)

**문제**: `_detect_device()` / `get_device()` / `_detect_gpu()` 가 5곳에서 다른 로직으로 분산.

### Tasks

#### 4-1. `rag_bench/utils/__init__.py` 생성

#### 4-2. `rag_bench/utils/device.py` 신설
```python
def detect_device(prefer_mps: bool = False) -> str:
    """CUDA > (MPS) > CPU 순으로 디바이스 자동 감지."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA: {name}")
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        print("[Device] MPS")
        return "mps"
    print("[Device] CPU")
    return "cpu"
```

#### 4-3. 기존 분산 감지 코드를 `detect_device()` 호출로 교체
- `strategies/colbert.py:94-100`
- `strategies/colbert_rerank.py:97-103`
- `strategies/dense_sparse.py:379` (`"cpu"` 하드코딩 → `detect_device()`)
- `run_tracker.py:91-101` `_detect_gpu()` → `detect_device()` 활용
- `colab_config.py:49-61` `get_device()` → `rag_bench.utils.device.detect_device` 재수출

**검증 기준**:
- `from rag_bench.utils.device import detect_device` 성공
- `grep -r "_detect_device\|get_device" rag_bench/strategies/` → 결과 없음

---

## Phase 5 — 중복 유틸리티 공유 모듈 추출 (MAJOR)

**문제**: `_load_qa_dataset`, 리포트 생성, 답변 생성 로직이 2-3곳에 복사되어 있음.

### Tasks

#### 5-1. `rag_bench/utils/qa_loader.py` 신설
**이동 대상** (`run_bench.py:29-39`, `run_all_combos.py:305-313` → 공유 모듈):
```python
def load_qa_dataset(path: Path) -> dict:
    """QA 데이터셋 JSON 로드. qa_pairs 리스트 포함 dict 반환."""
    ...
```

#### 5-2. `rag_bench/utils/report.py` 신설
**이동 대상** (`run_all_combos.py:926-1041`, `colab_runner.py:855-980`):
- `generate_markdown_report(results, config) -> str` 공통 함수
- 스크립트별 차이는 `extra_sections` 파라미터로 주입

#### 5-3. `_print_ragas_table()` 공유
**이동 대상** (`run_bench.py:78-104`, `run_all_combos.py:448-488` → `utils/report.py`):
```python
def print_ragas_table(reports: dict) -> None:
    ...
```

#### 5-4. 답변 생성 로직 공유
**이동 대상** (`runner.py:228-258`, `colab_runner.py:817-853`):
```python
# rag_bench/utils/answer_generator.py
def generate_answers(queries, context_docs, model: str, max_workers: int = 8) -> list:
    ...
```

**검증 기준**:
- `run_bench.py`, `run_all_combos.py` 에서 `_load_qa_dataset` 정의가 사라짐
- `from rag_bench.utils.qa_loader import load_qa_dataset` 성공

---

## Phase 6 — LLM 모델명 / 설정 중앙화 (MINOR)

**문제**: 모델명 4곳 하드코딩, `cleanup()` 시그니처 비일관.

### Tasks

#### 6-1. `config.py` LLM 상수 추가
```python
# rag_bench/config.py 추가
DEFAULT_ANSWER_LLM = "gpt-4o-mini"      # 답변 생성용
DEFAULT_EVAL_LLM   = "gpt-4o-mini"      # RAGAS 평가용
DEFAULT_CONTEXTUAL_LLM = "gpt-4o-mini"  # Contextual Retrieval 압축용
```

#### 6-2. 하드코딩 모델명을 상수로 교체
- `runner.py:62`: `"gpt-3.5-turbo"` → `config.DEFAULT_ANSWER_LLM`
- `strategies/contextual_retrieval.py:97`: `"gpt-4o-mini"` → `config.DEFAULT_CONTEXTUAL_LLM`
- `evaluation/evaluator.py:284`: `"gpt-4o-mini"` → `config.DEFAULT_EVAL_LLM`
- `colab_runner.py:562,824`: `"gpt-4o-nano"` → `colab_config.DEFAULT_COLAB_LLM` (별도 오버라이드)

#### 6-3. `cleanup()` 시그니처 일관화
**대상**: `DenseSparseStrategy.cleanup(delete_index: bool = False)` →
`BaseRAGStrategy.cleanup(**kwargs)` 또는 `DenseSparseStrategy.cleanup_index()` 별도 메서드로 분리.

#### 6-4. `graph/state.py` 네이밍 snake_case 통일 (MINOR)
- `questionIsClear` → `question_is_clear`
- `originalQuery` → `original_query`
- `rewrittenQuestions` → `rewritten_questions`

**검증 기준**:
- `grep -r '"gpt-3.5-turbo"\|"gpt-4o-mini"\|"gpt-4o-nano"' rag_bench/` → `config.py` 1곳만
- `python -c "from rag_bench.config import DEFAULT_EVAL_LLM"` 성공

---

## 의존성 그래프 (Phase 순서)

```
Phase 4 (utils/device.py 신설)
    ↓
Phase 2 (DenseSparseStrategy device 파라미터) — Phase 4 필요
    ↓
Phase 1 (combo/ 패키지 추출) — Phase 2 이후 캐시/빌더 안정화
    ↓
Phase 3 (StrategyRetriever) — Phase 1 완료 후 안전하게 진행
    ↓
Phase 5 (공유 유틸리티) — Phase 1 완료 후
    ↓
Phase 6 (상수 중앙화) — 언제든 독립 진행 가능
```

---

## 영향 범위 요약

| Phase | 수정 파일 수 | 위험도 | 예상 작업 크기 |
|-------|------------|--------|--------------|
| 1 (combo/ 추출) | ~8개 | 중 (import 경로 변경) | 大 |
| 2 (monkey-patch 제거) | ~4개 | 高 (Colab 동작 변경) | 中 |
| 3 (StrategyRetriever) | ~6개 | 低 (내부 변경) | 小 |
| 4 (device 일원화) | ~6개 | 低 | 小 |
| 5 (유틸리티 추출) | ~6개 | 中 | 中 |
| 6 (상수 중앙화) | ~5개 | 低 | 小 |

---

## 실행 우선순위 제안

```
🔴 즉시: Phase 3, 4, 6 — 위험도 낮고 효과 즉각
🟡 단기: Phase 1 — 가장 큰 구조 개선, 신중하게
🟡 단기: Phase 5 — 중복 제거
🟢 중기: Phase 2 — Colab 검증 필요, 충분한 테스트 후
```
