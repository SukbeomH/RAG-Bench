# 리팩토링 Phase 1 / 5 / 2 실행 플랜

## Date: 2026-02-20
## 선행 완료: Phase 3 (StrategyRetriever) / Phase 4 (detect_device) / Phase 6 (LLM 상수)

---

## 목표 역추적 (Goal-Backward)

| 최종 목표 | 검증 기준 |
|-----------|-----------|
| `run_all_combos.py` 1,348줄 God Script 분해 | Phase 1 완료 후 파일 < 750줄 |
| 중복 유틸리티 제거 | `_load_qa_dataset` 1곳, `_print_ragas_table` 1곳 |
| monkey-patch 제거 | `colab_config.py`에 `DenseSparseStrategy._init_*` 직접 할당 없음 |
| 전체 테스트 통과 | `python -c "from rag_bench.combo import ComboSpec, CacheConfig"` 성공 |

---

## Phase 1: `rag_bench/combo/` 패키지 추출

### 목표
`run_all_combos.py` 내 도메인 모델 / 캐시 관리 / 팩토리 함수를 별도 패키지로 분리하여
`colab_runner.py`와 `run_all_combos.py` 모두 동일 소스를 참조.

### 추출 대상 (원본 라인 번호)

| 심볼 | 원본 위치 | 이동 대상 |
|------|-----------|-----------|
| `ComboSpec` | `run_all_combos.py:59-93` | `combo/spec.py` |
| `PRESETS` | `run_all_combos.py:99-118` | `combo/spec.py` |
| `generate_valid_combinations()` | `run_all_combos.py:121-133` | `combo/spec.py` |
| `CacheConfig` | `run_all_combos.py:141-151` | `combo/cache.py` |
| `IndexCacheManager` | `run_all_combos.py:158-284` | `combo/cache.py` |
| `build_strategy_from_spec()` | `run_all_combos.py:292-332` | `combo/builder.py` |

> **주의**: `_try_build_*` 레거시 헬퍼 함수들(lines 351-432)은 `run_all_combos.py` 내부에서만 사용하므로 이동하지 않음.

### 태스크 목록

#### T1-1. `rag_bench/combo/` 디렉터리 + `__init__.py` 생성
```
rag_bench/combo/__init__.py  ← 공개 API re-export
rag_bench/combo/spec.py
rag_bench/combo/cache.py
rag_bench/combo/builder.py
```

`__init__.py` 내용:
```python
from rag_bench.combo.spec import ComboSpec, PRESETS, generate_valid_combinations
from rag_bench.combo.cache import CacheConfig, IndexCacheManager
from rag_bench.combo.builder import build_strategy_from_spec

__all__ = [
    "ComboSpec", "PRESETS", "generate_valid_combinations",
    "CacheConfig", "IndexCacheManager",
    "build_strategy_from_spec",
]
```

#### T1-2. `combo/spec.py` 작성
- `ComboSpec`, `PRESETS`, `generate_valid_combinations()` 이동
- 의존성: `rag_bench.strategies.dense_sparse.DENSE_MODELS, SPARSE_TYPES` (import)
- `Optional` typing import 필요

```python
"""ComboSpec — 3-Layer 조합 명세 + 프리셋."""
from dataclasses import dataclass
from typing import Dict, List, Optional
from rag_bench.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES

@dataclass
class ComboSpec:
    ...  # 그대로 복사

PRESETS: Dict[str, Dict[str, list]] = { ... }  # 그대로 복사

def generate_valid_combinations(config: Dict[str, list]) -> List[ComboSpec]:
    ...  # 그대로 복사
```

#### T1-3. `combo/cache.py` 작성
- `CacheConfig`, `IndexCacheManager` 이동
- 의존성:
  - `from rag_bench.config import BENCH_DATA_DIR`
  - `from rag_bench.combo.spec import ComboSpec`
  - `from rag_bench.config import DEFAULT_CONTEXTUAL_LLM` (contextual_llm 기본값)
- `CacheConfig.contextual_llm` 기본값: 하드코딩 `"gpt-4o-mini"` → `DEFAULT_CONTEXTUAL_LLM` 상수로 교체

```python
from rag_bench.config import BENCH_DATA_DIR, DEFAULT_CONTEXTUAL_LLM

@dataclass
class CacheConfig:
    colbert_model: str = "jinaai/jina-colbert-v2"
    colbert_device: str = "cpu"
    flashrank_model: str = "ms-marco-MultiBERT-L-12"
    flashrank_max_length: int = 512
    contextual_llm: str = DEFAULT_CONTEXTUAL_LLM  # ← 상수 참조
    rerank_n: int = 20
```

#### T1-4. `combo/builder.py` 작성
- `build_strategy_from_spec()` 이동
- 의존성:
  - `from rag_bench.combo.spec import ComboSpec`
  - `from rag_bench.combo.cache import IndexCacheManager`

#### T1-5. `run_all_combos.py` import 업데이트
추출된 심볼들의 정의를 제거하고 `combo` 패키지에서 import:
```python
from rag_bench.combo import (
    ComboSpec, PRESETS, generate_valid_combinations,
    CacheConfig, IndexCacheManager, build_strategy_from_spec,
)
```

#### T1-6. `colab_config.py` import 업데이트
```python
# Before
from rag_bench.scripts.run_all_combos import CacheConfig

# After
from rag_bench.combo import CacheConfig
```

#### T1-7. `colab_runner.py` import 업데이트
`colab_runner.py`에서 `run_all_combos` 내 심볼을 import하는 부분 → `rag_bench.combo`로 변경.

#### T1-8. `_try_build_contextual()` LLM 상수 수정
`run_all_combos.py:415` 하드코딩 `"gpt-4o-mini"` → `DEFAULT_CONTEXTUAL_LLM`:
```python
from rag_bench.config import DEFAULT_CONTEXTUAL_LLM
# ...
strategy = ContextualRetrievalStrategy(
    base_strategy=base,
    parent_pairs=parent_pairs,
    llm_model=DEFAULT_CONTEXTUAL_LLM,  # ← 상수 참조
)
```

### 검증 기준
```python
python3 -c "from rag_bench.combo import ComboSpec, CacheConfig, IndexCacheManager, build_strategy_from_spec; print('OK')"
python3 -c "from rag_bench.scripts.run_all_combos import PRESETS, generate_valid_combinations; print('OK')"
python3 -c "from rag_bench_colab.colab_config import get_cache_config; print('OK')"
```

### 커밋
```
refactor(combo): ComboSpec/CacheConfig/IndexCacheManager → rag_bench/combo/ 패키지 추출
```

---

## Phase 5: 공유 유틸리티 추출

### 목표
2곳 이상 중복된 유틸리티 함수를 `rag_bench/utils/` 하위 모듈로 통합.

### 추출 대상

| 함수 | 중복 위치 | 이동 대상 |
|------|-----------|-----------|
| `_load_qa_dataset()` | `run_bench.py:29-39`, `run_all_combos.py:340-348` | `utils/qa_loader.py` |
| `_print_ragas_table()` | `run_bench.py:78-104` (간단), `run_all_combos.py:483-523` (확장) | `utils/report.py` |

> `colab_runner.py:857`의 `_generate_report()`는 Colab 세션 전용 상태(`self.session_id`, `self.preset` 등)에 의존하므로 **이번 Phase에서 추출하지 않음**.

### 태스크 목록

#### T5-1. `rag_bench/utils/qa_loader.py` 작성

두 파일의 `_load_qa_dataset()`은 동일한 로직이지만 `BENCH_DATA_DIR` import 경로만 다름.
공통 버전 작성:

```python
"""QA 데이터셋 로드 유틸리티."""
import json
import sys
from pathlib import Path

def load_qa_dataset(data_dir: Path) -> dict:
    """qa_dataset.json을 로드한다.

    Args:
        data_dir: _benchdata 디렉터리 경로 (BENCH_DATA_DIR).

    Returns:
        { 'num_qa': int, 'qa_pairs': List[dict] } 형태의 dict.
    """
    qa_path = data_dir / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)
    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset
```

#### T5-2. `rag_bench/utils/__init__.py` 업데이트
기존 `utils/__init__.py`에 `qa_loader` re-export 추가:
```python
from rag_bench.utils.qa_loader import load_qa_dataset
```

#### T5-3. `rag_bench/utils/report.py` 작성

`run_all_combos.py`의 확장 버전(`_print_ragas_table`)을 기준으로 작성
(scoring profile 지원 버전이 더 기능이 많으므로 이를 공용으로 채택):

```python
"""RAGAS 결과 출력 유틸리티."""
from typing import Optional
import pandas as pd

def print_ragas_table(scores_df: Optional[pd.DataFrame], scoring_profile: str = "balanced") -> None:
    """RAGAS 평가 결과를 콘솔에 포맷팅하여 출력한다.

    Args:
        scores_df: 전략별 메트릭 DataFrame.
        scoring_profile: SCORING_PROFILES 키 (balanced, precision_critical 등).
    """
    if scores_df is None or scores_df.empty:
        print("RAGAS 평가 결과가 없습니다.")
        return

    from rag_bench.evaluation.evaluator import SCORING_PROFILES

    print(f"\n{'═' * 100}")
    print(f" RAGAS 평가 결과 비교 (scoring: {scoring_profile})")
    print(f"{'═' * 100}")

    # ... (run_all_combos.py:483-523 로직 그대로)
```

#### T5-4. `run_bench.py` 업데이트
```python
# Before
def _load_qa_dataset() -> dict: ...
def _print_ragas_table(scores_df): ...

# After
from rag_bench.config import BENCH_DATA_DIR
from rag_bench.utils.qa_loader import load_qa_dataset
from rag_bench.utils.report import print_ragas_table

# main() 내에서
dataset = load_qa_dataset(BENCH_DATA_DIR)
print_ragas_table(scores_df)
```

#### T5-5. `run_all_combos.py` 업데이트
```python
# Before (lines 340-348)
def _load_qa_dataset() -> dict: ...

# After
from rag_bench.utils.qa_loader import load_qa_dataset
from rag_bench.utils.report import print_ragas_table

# 호출부: _load_qa_dataset() → load_qa_dataset(BENCH_DATA_DIR)
# 호출부: _print_ragas_table(df) → print_ragas_table(df)
```

> `run_all_combos.py`의 `_print_ragas_table` 정의(lines 483-523)는 `utils/report.py`로 이동 후 삭제.

### 검증 기준
```python
python3 -c "from rag_bench.utils.qa_loader import load_qa_dataset; print('OK')"
python3 -c "from rag_bench.utils.report import print_ragas_table; print('OK')"
python3 -c "from rag_bench.scripts.run_bench import main; print('OK')"
python3 -c "from rag_bench.scripts.run_all_combos import _run_preset_mode; print('OK')"
```

### 커밋
```
refactor(utils): load_qa_dataset / print_ragas_table 공유 모듈 추출
```

---

## Phase 2: monkey-patch → DI 패턴

### 목표
`colab_config.py`의 세 monkey-patch 함수를 제거하고, 생성자 파라미터 / 환경 변수 기반으로 대체.

### 패치 목록 분석

| 패치 함수 | 현재 동작 | 대안 |
|-----------|-----------|------|
| `patch_dense_device(device)` | `DenseSparseStrategy._init_dense` 메서드 교체 | 생성자에 `device` 파라미터 추가 |
| `patch_rag_bench_config()` | `rag_bench.config` 모듈 속성 직접 변경 | 환경변수 기반 config 오버라이드 |
| `patch_qdrant_memory_mode()` | `DenseSparseStrategy._init_qdrant` 메서드 교체 | `_init_qdrant` 내부에서 `:memory:` 분기 처리 |

### 태스크 목록

#### T2-1. `DenseSparseStrategy.__init__()` — `device` 파라미터 추가

`dense_sparse.py`의 `DenseSparseStrategy.__init__()` 시그니처 변경:
```python
def __init__(
    self,
    dense_model: Optional[str] = None,
    sparse_type: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    combo_id: Optional[int] = None,
    device: Optional[str] = None,   # ← 신규: None이면 detect_device() 자동 사용
):
    ...
    self._device = device  # None → _init_dense에서 detect_device() 호출
```

`_init_dense()` 수정:
```python
def _init_dense(self):
    ...
    else:
        # HuggingFace 로컬 모델
        from langchain_huggingface import HuggingFaceEmbeddings
        _device = self._device if self._device is not None else detect_device()
        self._dense_embeddings = HuggingFaceEmbeddings(
            model_name=model_spec,
            model_kwargs={"device": _device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
```

#### T2-2. `DenseSparseStrategy._init_qdrant()` — `:memory:` 모드 내장

`_init_qdrant()` 내에 `:memory:` 분기 추가 (`patch_qdrant_memory_mode`의 로직 흡수):
```python
def _init_qdrant(self):
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.qdrant import RetrievalMode

    if self._client is None:
        if self._qdrant_path == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=self._qdrant_path)

    if not self._client.collection_exists(self._collection_name):
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
    ...
```

#### T2-3. `rag_bench/config.py` — 환경변수 오버라이드 지원

`config.py` 하단에 환경변수 기반 오버라이드 블록 추가:
```python
import os as _os

# 환경변수 오버라이드 (Colab 등 외부 환경에서 경로 변경)
if _raw_data := _os.environ.get("RAG_BENCH_DATA_DIR"):
    BENCH_DATA_DIR = Path(_raw_data)
if _raw_docs := _os.environ.get("RAG_BENCH_DOCS_DIR"):
    BENCH_DOCS_DIR = Path(_raw_docs)
if _raw_docs_src := _os.environ.get("RAG_BENCH_DOCS_SRC"):
    DOCS_DIR = Path(_raw_docs_src)
```

#### T2-4. `colab_config.py` 업데이트

`patch_rag_bench_config()` 내부에서 모듈 속성 직접 변경 대신 환경변수 설정으로 교체:
```python
def patch_rag_bench_config(qdrant_mode: str = "ephemeral") -> None:
    """Colab 경로를 환경변수로 설정하여 rag_bench.config가 자동 오버라이드되게 한다."""
    import os
    os.environ["RAG_BENCH_DATA_DIR"] = str(DRIVE_BENCHDATA_DIR)
    os.environ["RAG_BENCH_DOCS_DIR"] = str(COLAB_DOCS_DIR)
    os.environ["RAG_BENCH_DOCS_SRC"] = str(COLAB_PDF_DIR)

    # config 모듈을 아직 import하지 않았다면 환경변수만으로 충분
    # 이미 import된 경우 모듈 속성도 업데이트 (하위 호환)
    try:
        import rag_bench.config as cfg
        cfg.BENCH_DATA_DIR = DRIVE_BENCHDATA_DIR
        cfg.BENCH_DOCS_DIR = COLAB_DOCS_DIR
        cfg.DOCS_DIR = COLAB_PDF_DIR
        cfg.MODELS_DIR = DRIVE_MODELS_DIR
    except ImportError:
        pass

    print(f"[Config] RAG_BENCH_DATA_DIR → {DRIVE_BENCHDATA_DIR}")
    print(f"[Config] RAG_BENCH_DOCS_DIR → {COLAB_DOCS_DIR}")
```

`patch_dense_device()` 교체: Colab에서 `IndexCacheManager` 생성 시 `device` 파라미터 전달로 대체.
→ `get_cache_config(device)` 헬퍼에 문서화:
```python
def get_cache_config(device: str = "cuda") -> "CacheConfig":
    """Colab 환경에 맞는 CacheConfig를 반환한다.

    DenseSparseStrategy에 device를 직접 전달하므로 monkey-patch 불필요.
    IndexCacheManager.get_or_build()에서 device 파라미터가 DenseSparseStrategy로 전달됨.
    """
    from rag_bench.combo import CacheConfig
    return CacheConfig(colbert_device=device)
```

`patch_qdrant_memory_mode()` 제거 (T2-2에서 내장 처리됨).

`init_colab()` 업데이트:
```python
def init_colab(qdrant_mode="ephemeral", device=None, mount_drive=True) -> dict:
    info = setup_colab_env(mount_drive=mount_drive)
    if device is None:
        device = info["device"]

    patch_rag_bench_config(qdrant_mode=qdrant_mode)
    _patch_hf_hub_for_colab()
    # patch_dense_device(device) ← 제거 (DenseSparseStrategy 생성자로 대체)
    # patch_qdrant_memory_mode() ← 제거 (_init_qdrant 내장)
    _setup_korean_font()

    info["qdrant_mode"] = qdrant_mode
    info["patched"] = True
    return info
```

#### T2-5. `IndexCacheManager.get_or_build()` — `device` 전달

`combo/cache.py`의 `get_or_build()` 메서드에서 `DenseSparseStrategy` 생성 시 `device` 전달:
```python
def get_or_build(self, spec: ComboSpec, child_chunks, reindex=False):
    ...
    strategy = DenseSparseStrategy(
        dense_model=spec.dense,
        sparse_type=spec.sparse,
        qdrant_path=qdrant_path,
        device=self.config.colbert_device or None,  # CacheConfig에서 device 전달
    )
```

> **주의**: `CacheConfig.colbert_device`는 ColBERT용이므로 Dense 임베딩 디바이스와 혼용 주의.
> `CacheConfig`에 `dense_device: str = "cpu"` 필드 추가를 검토하거나, `colbert_device`를 Dense에도 재사용.

### 검증 기준
```python
# monkey-patch 함수가 더 이상 _init_* 메서드를 교체하지 않음을 확인
python3 -c "
from rag_bench_colab.colab_config import init_colab
import rag_bench.strategies.dense_sparse as ds
orig = ds.DenseSparseStrategy._init_dense
init_colab.__doc__  # monkey-patch 없이 초기화
assert ds.DenseSparseStrategy._init_dense is orig, 'monkey-patch 미제거!'
print('OK: _init_dense 원본 유지됨')
"

# :memory: 모드 내장 확인
python3 -c "
from rag_bench.strategies.dense_sparse import DenseSparseStrategy
s = DenseSparseStrategy(dense_model='minilm', sparse_type='fastembed_bm25', qdrant_path=':memory:')
print('OK: :memory: 파라미터 수용됨')
"
```

### 커밋
```
refactor(dense): DenseSparseStrategy device/memory 파라미터 내장 — monkey-patch 제거
refactor(colab): patch_dense_device/patch_qdrant_memory_mode 제거 — DI 패턴 적용
```

---

## 실행 순서 (의존성)

```
Phase 1 (combo/ 패키지)
  ↓ (colab_config가 CacheConfig를 combo에서 import)
Phase 5 (utils 공유 모듈)
  ↓ (run_all_combos.py import 경로 정리 후)
Phase 2 (monkey-patch → DI)
```

## 리스크 & 주의사항

1. **Phase 1**: `run_all_combos.py`에서 정의를 제거 후 `import`로 대체 시
   `from rag_bench.scripts.run_all_combos import ComboSpec` 같은 외부 사용처 없는지 확인.
   현재 확인된 외부 사용: `colab_config.py:387` (`CacheConfig`만)

2. **Phase 2**: `patch_dense_device`를 제거하면 `init_colab()` 호출 코드가 많은 Colab 노트북에 영향.
   → `patch_dense_device()`를 deprecated warning과 함께 유지하되 no-op 처리하는 전환 기간 운영 고려.

3. **Phase 2**: `CacheConfig.colbert_device`를 Dense 임베딩에도 재사용하면 의미가 혼용됨.
   → `CacheConfig`에 `dense_device: Optional[str] = None` 필드 별도 추가 권장.

4. **Phase 5**: `run_bench.py`의 `_print_ragas_table`은 단순 버전이므로
   `utils/report.py`의 확장 버전(scoring profile 지원)으로 교체 시 출력 형식이 변경됨 — 허용 가능.

---

## 완료 체크리스트

### Phase 1
- [ ] `rag_bench/combo/__init__.py` 생성
- [ ] `rag_bench/combo/spec.py` 생성 (ComboSpec, PRESETS, generate_valid_combinations)
- [ ] `rag_bench/combo/cache.py` 생성 (CacheConfig, IndexCacheManager) + DEFAULT_CONTEXTUAL_LLM 참조
- [ ] `rag_bench/combo/builder.py` 생성 (build_strategy_from_spec)
- [ ] `run_all_combos.py` 정의 제거 → combo import
- [ ] `run_all_combos.py:415` `"gpt-4o-mini"` → `DEFAULT_CONTEXTUAL_LLM`
- [ ] `colab_config.py:387` import 경로 변경
- [ ] `colab_runner.py` import 경로 변경
- [ ] 검증 명령 통과

### Phase 5
- [ ] `rag_bench/utils/qa_loader.py` 생성
- [ ] `rag_bench/utils/report.py` 생성
- [ ] `rag_bench/utils/__init__.py` re-export 추가
- [ ] `run_bench.py` import 업데이트 + 로컬 함수 제거
- [ ] `run_all_combos.py` import 업데이트 + 로컬 함수 제거
- [ ] 검증 명령 통과

### Phase 2
- [ ] `DenseSparseStrategy.__init__()` — `device` 파라미터 추가
- [ ] `DenseSparseStrategy._init_dense()` — `self._device` 반영
- [ ] `DenseSparseStrategy._init_qdrant()` — `:memory:` 분기 내장
- [ ] `rag_bench/config.py` — 환경변수 오버라이드 블록 추가
- [ ] `colab_config.py` — `patch_dense_device` 제거 또는 deprecated 처리
- [ ] `colab_config.py` — `patch_qdrant_memory_mode` 제거
- [ ] `colab_config.py` — `patch_rag_bench_config` 환경변수 방식으로 교체
- [ ] `combo/cache.py` `get_or_build()` — device 전달
- [ ] 검증 명령 통과
