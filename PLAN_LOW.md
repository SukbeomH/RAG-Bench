# PLAN_LOW.md — 낮은 우선순위 기술 부채 해결

## 배경

현재 코드베이스에는 다음 7개의 낮은 우선순위 기술 부채가 존재한다:

| 번호 | 항목 | 파일 |
|------|------|------|
| #11 | `psutil` 미명시 + Linux 폴백 없음 | `pyproject.toml`, `rag_bench/run_tracker.py` |
| #12 | `PROJECT_ROOT` / `PACKAGE_ROOT` 혼재, `QDRANT_BASE_PATH` 불일치 | `rag_bench/config.py` |
| #13 | 타입 힌트 `Any` 미구체화 | `rag_bench/base.py:103`, `rag_bench/strategies/dense_sparse.py:260-262` |
| #14 | 노트북 로컬 경로 하드코딩 | `rag_bench/scripts/bench_visualize.ipynb` |
| #15 | `_MultiPerspectiveLLM` 뮤터블 기본 인수 `callbacks: List[Any] = []` | `rag_bench/evaluation/evaluator.py` |
| #17 | `IndexCacheManager` `@dataclass` 프라이빗 필드 외부 노출 | `rag_bench/combo/cache.py` |
| #18 | `RunTracker` 프라이빗 속성 직접 접근 | `rag_bench_colab/colab_runner.py`, `rag_bench/run_tracker.py` |

---

## 의존성 분석

```
#18 (RunTracker 공개 API) ─────────┐
                                   ├──→ #17 (IndexCacheManager) 독립
#15 (뮤터블 기본 인수) ──────────── 독립
#13 (타입 힌트) ────────────────── 독립
#12 (config.py 경로 정리) ─────────┐
                                   └──→ #14 (노트북 경로) 는 #12 이후 수행
#11 (psutil 의존성) ──────────── 독립
```

- **#12** 는 경로 상수를 정비하므로 **#14** (노트북이 경로 상수를 참조하도록 변경) 이전에 수행해야 한다.
- **#18** 은 `RunTracker`에 공개 API를 추가한 뒤 `colab_runner.py`를 수정하므로 단일 Phase 내에서 순차 처리한다.
- 나머지 항목은 서로 독립적이다.

---

## Phase 구성

### Phase 1 — 안전한 버그 픽스 (뮤터블 기본 인수)

- **Task 1.1**: `evaluator.py`에서 뮤터블 기본 인수 `callbacks: List[Any] = []` 를 `None`으로 교체 (#15)

### Phase 2 — 의존성 명시 + 플랫폼 폴백 (#11)

- **Task 2.1**: `pyproject.toml`에 `psutil` 명시적 의존성 추가
- **Task 2.2**: `run_tracker.py`의 `collect_platform_info()`에 Linux `/proc/meminfo` 폴백 추가

### Phase 3 — config.py 경로 상수 정비 (#12)

- **Task 3.1**: `config.py`에서 사용되지 않는 `QDRANT_BASE_PATH` 제거 및 `PROJECT_ROOT` 기준 경로를 `BENCH_DATA_DIR` 기준으로 통합
- **Task 3.2**: `PARENT_STORE_PATH`를 `BENCH_DATA_DIR / "parent_store"`로 이동, 기존 참조 업데이트

### Phase 4 — 타입 힌트 구체화 (#13) + 노트북 경로 (#14)

- **Task 4.1**: `base.py:103`의 `strategy: Any` 를 `BaseRAGStrategy`로 구체화
- **Task 4.2**: `dense_sparse.py:260-262`의 `_dense_embeddings`, `_sparse_embeddings`, `_vector_store` 타입 힌트 구체화
- **Task 4.3**: `bench_visualize.ipynb`에서 하드코딩된 경로를 `config.py`의 `BENCH_DATA_DIR`로 교체 (#14)

### Phase 5 — 캡슐화 개선 (#17, #18)

- **Task 5.1**: `run_tracker.py`의 `RunTracker`에 `get_snapshot()` 공개 메서드 추가 (#18)
- **Task 5.2**: `colab_runner.py`의 프라이빗 속성 직접 접근을 `get_snapshot()` 호출로 교체 (#18)
- **Task 5.3**: `combo/cache.py`의 `IndexCacheManager`를 일반 클래스(`__init__` 초기화)로 변경, `@dataclass` 제거 (#17)

---

## 체크리스트

### Phase 1 — 뮤터블 기본 인수 수정

- [ ] **Task 1.1** — `evaluator.py` 뮤터블 기본 인수 교체 (#15)
  - files: `rag_bench/evaluation/evaluator.py`
  - action:
    1. `agenerate_text()` 시그니처를 `callbacks: Optional[List[Any]] = None` 으로 변경
    2. 함수 본문 첫 줄에 `callbacks = callbacks or []` 추가
    3. `generate_text()` 시그니처 동일하게 변경, 본문에 `callbacks = callbacks or []` 추가
    4. `generate()` 시그니처 동일하게 변경, 본문에 `callbacks = callbacks or []` 추가
  - verify: `rg "callbacks.*=.*\[\]" rag_bench/evaluation/evaluator.py` 결과가 0건
  - done: `evaluator.py` 내 모든 `callbacks` 파라미터가 `Optional[List[Any]] = None`

### Phase 2 — psutil 의존성 + 폴백

- [ ] **Task 2.1** — `pyproject.toml`에 `psutil` 추가 (#11)
  - files: `pyproject.toml`
  - action: `dependencies` 목록에 `"psutil>=5.9"` 추가 (알파벳 순서 유지)
  - verify: `rg "psutil" pyproject.toml` 결과가 1건 이상
  - done: `uv pip install -e .` 에서 `psutil`이 명시적으로 설치됨

- [ ] **Task 2.2** — `run_tracker.py` Linux 폴백 추가 (#11)
  - files: `rag_bench/run_tracker.py`
  - action: `collect_platform_info()` 함수의 `except ImportError` 블록에서, macOS `sysctl` 분기 아래에 Linux `/proc/meminfo` 폴백 분기를 추가:
    ```python
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = round(kb / (1024 ** 2), 1)
                    elif line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        info["ram_available_gb"] = round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    ```
  - verify: `rg "/proc/meminfo" rag_bench/run_tracker.py` 결과가 1건
  - done: `collect_platform_info()` 내에 Darwin, Linux 두 폴백 경로가 모두 존재

### Phase 3 — config.py 경로 상수 정비

- [ ] **Task 3.1** — 사용되지 않는 경로 상수 정리 (#12)
  - files: `rag_bench/config.py`
  - action:
    1. `QDRANT_BASE_PATH = PROJECT_ROOT` 제거 — 실제 Qdrant 인덱스는 `BENCH_DATA_DIR` 하위에 생성되므로 혼란 유발
    2. `MARKDOWN_DIR = PROJECT_ROOT / "markdown"` 를 `BENCH_DATA_DIR / "markdown"` 으로 변경
    3. 주석 섹션 "디렉토리 경로"와 "패키지 내부 경로" 를 하나로 통합하여 기준점 명확히 문서화
    4. 프로젝트 전체에서 `QDRANT_BASE_PATH` 잔여 참조 제거
  - verify:
    - `rg "QDRANT_BASE_PATH" rag_bench/` 결과가 0건
    - `python -c "from rag_bench.config import BENCH_DATA_DIR, MARKDOWN_DIR; print('OK')"` 성공
  - done: `config.py`에 `QDRANT_BASE_PATH`가 존재하지 않으며, 경로 상수 기준이 명확히 분리됨

- [ ] **Task 3.2** — `PARENT_STORE_PATH` 이전 (#12)
  - files: `rag_bench/config.py`, `PARENT_STORE_PATH`를 참조하는 모든 파일
  - action:
    1. `PARENT_STORE_PATH = PROJECT_ROOT / "parent_store"` 를 `PARENT_STORE_PATH = BENCH_DATA_DIR / "parent_store"` 로 변경
    2. `ensure_dirs()` 함수에서 `PARENT_STORE_PATH` 생성을 유지
    3. `rg "PARENT_STORE_PATH" rag_bench/` 로 모든 참조를 확인하고 동작 영향 검토
  - verify: `python -c "from rag_bench.config import PARENT_STORE_PATH; assert '_benchdata' in str(PARENT_STORE_PATH); print('OK')"` 성공
  - done: `PARENT_STORE_PATH`가 `_benchdata/parent_store`를 가리킴

### Phase 4 — 타입 힌트 + 노트북 경로

- [ ] **Task 4.1** — `base.py` 타입 힌트 구체화 (#13)
  - files: `rag_bench/base.py`
  - action: L103의 `strategy: Any` 를 `strategy: "BaseRAGStrategy"` (forward reference 문자열)로 변경. `Any`가 다른 곳에서 쓰이지 않으면 임포트 제거
  - verify: `rg "strategy: Any" rag_bench/base.py` 결과가 0건
  - done: `StrategyRetriever.strategy` 필드 타입이 `BaseRAGStrategy` (forward ref)

- [ ] **Task 4.2** — `dense_sparse.py` 타입 힌트 구체화 (#13)
  - files: `rag_bench/strategies/dense_sparse.py`
  - action:
    1. `self._dense_embeddings: Any = None` → `self._dense_embeddings: Optional[Embeddings] = None` (`from langchain_core.embeddings import Embeddings` 추가)
    2. `self._sparse_embeddings: Any = None` → `Optional[Union[KoreanBM25Encoder, SpladeEncoder, "FastEmbedSparse"]] = None` (또는 주석으로 실제 타입 명시)
    3. `self._vector_store: Any = None` → `self._vector_store: Optional["QdrantVectorStore"] = None`
  - verify: `rg "_dense_embeddings: Any|_sparse_embeddings: Any|_vector_store: Any" rag_bench/strategies/dense_sparse.py` 결과가 0건
  - done: 3개 인스턴스 변수의 타입 힌트가 구체적 타입 또는 Union 타입

- [ ] **Task 4.3** — `bench_visualize.ipynb` 경로 하드코딩 제거 (#14)
  - files: `rag_bench/scripts/bench_visualize.ipynb`
  - action: `DATA_DIR = Path("../") / "_benchdata"` 를 다음으로 교체:
    ```python
    try:
        from rag_bench.config import BENCH_DATA_DIR
        DATA_DIR = BENCH_DATA_DIR
    except ImportError:
        DATA_DIR = Path("../") / "_benchdata"  # 폴백
    ```
  - verify: 노트북 소스에 `Path("../") / "_benchdata"` 가 폴백 분기에만 존재
  - done: 노트북이 `rag_bench.config.BENCH_DATA_DIR` 을 우선 사용하고, 불가 시에만 상대 경로로 폴백

### Phase 5 — 캡슐화 개선

- [ ] **Task 5.1** — `RunTracker`에 `get_snapshot()` 공개 메서드 추가 (#18)
  - files: `rag_bench/run_tracker.py`
  - action: `RunTracker` 클래스에 다음 메서드 추가:
    ```python
    def get_snapshot(self) -> dict:
        """현재까지의 기록을 dict로 반환한다 (시각화/외부 연동용).

        finalize() 호출 전에도 사용 가능하며, 이 시점까지의 데이터를 스냅샷으로 반환한다.
        """
        from dataclasses import asdict
        rec = self._record
        rec.strategy_timings = [asdict(t) for t in self._timings]
        rec.phase_times = [asdict(p) for p in self._phases]
        rec.token_usage_total = asdict(self._token_total)
        return asdict(rec)
    ```
  - verify: `python -c "from rag_bench.run_tracker import RunTracker; assert hasattr(RunTracker, 'get_snapshot'); print('OK')"` 성공
  - done: `RunTracker.get_snapshot()` 메서드가 존재하고 `dict`를 반환

- [ ] **Task 5.2** — `colab_runner.py` 프라이빗 접근 제거 (#18)
  - files: `rag_bench_colab/colab_runner.py`
  - action: `get_run_record()` 메서드를 다음으로 교체:
    ```python
    def get_run_record(self) -> Optional[dict]:
        """RunTracker의 현재 기록을 반환한다 (시각화용)."""
        if self._tracker is None:
            return None
        try:
            return self._tracker.get_snapshot()
        except Exception:
            return None
    ```
  - verify: `rg "_tracker\._record|_tracker\._timings|_tracker\._phases|_tracker\._token_total" rag_bench_colab/colab_runner.py` 결과가 0건
  - done: `colab_runner.py`가 `RunTracker`의 프라이빗 속성에 직접 접근하지 않음

- [ ] **Task 5.3** — `IndexCacheManager` `@dataclass` 제거 (#17)
  - files: `rag_bench/combo/cache.py`
  - action:
    1. `IndexCacheManager`에서 `@dataclass` 데코레이터 제거
    2. `__init__` 메서드를 명시적으로 정의:
       ```python
       class IndexCacheManager:
           """동일 (dense, sparse) 쌍은 같은 Qdrant 인덱스를 재사용."""

           def __init__(self, config: Optional[CacheConfig] = None):
               self.config = config or CacheConfig()
               self.cache: Dict[str, Tuple[Any, str]] = {}
               self.ctx_cache: Dict[str, Any] = {}
               self._colbert_model: Any = None
               self._flashrank_ranker: Any = None
       ```
    3. `CacheConfig`는 `@dataclass`를 유지하므로 `from dataclasses import dataclass, field` 임포트 유지
  - verify:
    - `rg "@dataclass" rag_bench/combo/cache.py` 결과에 `CacheConfig` 위만 존재 (`IndexCacheManager` 위 없음)
    - `python -c "from rag_bench.combo.cache import IndexCacheManager; m = IndexCacheManager(); print(m.cache); print('OK')"` 성공
  - done: `IndexCacheManager`가 일반 클래스이며, `_colbert_model`, `_flashrank_ranker`가 `dataclasses.fields()` 에 노출되지 않음

---

## 실행 Wave 상세

```
W1 (독립 버그픽스):
  Task 1.1  evaluator.py       뮤터블 기본 인수 수정
  Task 2.1  pyproject.toml     psutil 의존성 추가
  Task 2.2  run_tracker.py     Linux 폴백 추가
  → 수정 파일 겹침 없음, 병렬 실행 가능

W2 (config 경로 정비):
  Task 3.1  config.py          QDRANT_BASE_PATH 제거, 경로 통합
  Task 3.2  config.py          PARENT_STORE_PATH 이전
  → 동일 파일(config.py) 수정, 순차 실행

W3 (타입 + API 추가):
  Task 4.1  base.py            strategy 타입 힌트
  Task 4.2  dense_sparse.py    임베딩/벡터스토어 타입 힌트
  Task 4.3  bench_visualize.ipynb  경로 config 참조 (W2 이후)
  Task 5.1  run_tracker.py     get_snapshot() 추가
  → 4개 서로 다른 파일, 병렬 실행 가능

W4 (캡슐화 마무리):
  Task 5.2  colab_runner.py    프라이빗 접근 → get_snapshot() (W3 Task 5.1 이후)
  Task 5.3  combo/cache.py     @dataclass 제거
  → 서로 다른 파일, 병렬 실행 가능
```

---

## 주의사항

1. **Task 3.1 (QDRANT_BASE_PATH 제거)**: 프로젝트 전체에서 `QDRANT_BASE_PATH`를 참조하는 곳이 없는지 반드시 `rg`로 검증한다.
2. **Task 3.2 (PARENT_STORE_PATH 이전)**: 기존에 `PROJECT_ROOT/parent_store/`에 생성된 데이터가 있을 수 있다. 마이그레이션 안내를 MEMORY.md에 기록한다.
3. **Task 4.2 (Sparse 타입 힌트)**: `KoreanBM25Encoder`, `SpladeEncoder`, `FastEmbedSparse` 에 공통 프로토콜(ABC)이 없다. 이 단계에서는 `Union` 타입 또는 주석으로 실제 타입을 명시한다.
4. **Task 4.3 (노트북 경로)**: Jupyter 환경에서는 `__file__`이 정의되지 않으므로, `try/except ImportError` 패턴으로 폴백을 반드시 유지한다.
5. **Task 5.3 (IndexCacheManager)**: `@dataclass` 제거 시 생성자 시그니처 `IndexCacheManager(config=CacheConfig())` 형태를 유지한다.

---

## 수정 파일 요약

| 파일 | 관련 Task | 변경 내용 요약 |
|------|-----------|---------------|
| `pyproject.toml` | 2.1 | `psutil>=5.9` 의존성 추가 |
| `rag_bench/config.py` | 3.1, 3.2 | `QDRANT_BASE_PATH` 제거, `MARKDOWN_DIR`/`PARENT_STORE_PATH` 기준 변경 |
| `rag_bench/base.py` | 4.1 | `strategy: Any` → `strategy: "BaseRAGStrategy"` |
| `rag_bench/strategies/dense_sparse.py` | 4.2 | `_dense_embeddings`/`_sparse_embeddings`/`_vector_store` 타입 힌트 구체화 |
| `rag_bench/scripts/bench_visualize.ipynb` | 4.3 | `DATA_DIR` 상대 경로를 `BENCH_DATA_DIR` 임포트 + 폴백 구조로 교체 |
| `rag_bench/evaluation/evaluator.py` | 1.1 | `callbacks: List[Any] = []` 3곳을 `Optional[List[Any]] = None` + `or []` 패턴으로 교체 |
| `rag_bench/combo/cache.py` | 5.3 | `IndexCacheManager`에서 `@dataclass` 제거, 명시적 `__init__` 정의 |
| `rag_bench/run_tracker.py` | 2.2, 5.1 | Linux `/proc/meminfo` 폴백 추가, `get_snapshot()` 공개 메서드 추가 |
| `rag_bench_colab/colab_runner.py` | 5.2 | `_tracker._record` 등 프라이빗 접근을 `_tracker.get_snapshot()` 호출로 교체 |
