---
title: "기술 부채 5건 일괄 해소 — 공개 API, CacheConfig, 레거시 제거"
tags:
  - execution
  - summary
  - refactoring
  - tech-debt
  - CacheConfig
  - legacy-removal
  - parent-store
type: execution-summary
created: 2026-02-19T14:00:00Z
contextual_description: "IndexCacheManager 하드코딩 외부화(CacheConfig), 전략 클래스 공개 API 추가, RAGAS v0.3 레거시 삭제, parent_store 포맷 통일, 중복 pyproject.toml 제거 — 11 files, +124/-306"
keywords:
  - CacheConfig
  - IndexCacheManager
  - share_embeddings
  - DenseSparseStrategy
  - ColBERTRerankStrategy
  - FlashRankRerankStrategy
  - RAGEvaluator
  - legacy.py
  - parents.json
  - monkey-patch
  - patch_colbert_device
related:
  - 2026-02-19_colab-metricpreset-runtracker
  - 2026-02-19_colab-runner-optimization-sync
  - 2026-02-19_rag-bench-evaluation-upgrade
---

## 기술 부채 5건 일괄 해소 — 공개 API, CacheConfig, 레거시 제거

### Commit
- **Hash**: `0d3026e`
- **Branch**: `refactor/tech-debt-cleanup`
- **Files**: 11 changed, +124 / -306 (순감 182줄)

### Phase 1: Private 속성 직접 주입 해소 + IndexCacheManager 하드코딩 제거

**1-1. 전략 클래스 공개 API 추가**
- `DenseSparseStrategy.share_embeddings(dense, sparse, embedding_dim, use_langchain_sparse)` 메서드 신규 추가
  - `get_or_build_contextual()`에서 private 속성 직접 주입 대신 이 API 사용
- `ColBERTRerankStrategy`: 생성자에서 `shared_model` 전달 시 `_is_ready = True` 자동 설정
  - `build_strategy_from_spec()`에서 `strategy._is_ready = True` 직접 주입 제거
- `FlashRankRerankStrategy`: 동일하게 `shared_ranker` 전달 시 `_is_ready = True` 자동 설정

**1-2. CacheConfig dataclass 추출**
```python
@dataclass
class CacheConfig:
    colbert_model: str = "jinaai/jina-colbert-v2"
    colbert_device: str = "cpu"
    flashrank_model: str = "ms-marco-MultiBERT-L-12"
    flashrank_max_length: int = 512
    contextual_llm: str = "gpt-4o-mini"
    rerank_n: int = 20
```
- `IndexCacheManager.__init__`에 `config: CacheConfig` 파라미터 추가
- `get_colbert_model()`, `get_flashrank_ranker()`에서 config 참조
- `build_strategy_from_spec()`에서 `strategy._device = "cpu"` 하드코딩 제거 → `device=cfg.colbert_device`

**1-3. Colab monkey-patch 제거**
- `colab_config.py`: `patch_colbert_device()` 함수 전체 삭제 (56줄 감소)
- 대체: `get_cache_config(device)` → `CacheConfig(colbert_device=device)` 반환
- `colab_runner.py`: `IndexCacheManager(config=get_cache_config(self.device))` 형태로 변경
- `init_colab()`: `patch_colbert_device(device="cuda")` 호출 제거

### Phase 2: RAGAS v0.3 레거시 완전 제거
- `rag_bench/evaluation/legacy.py` 삭제 (136줄)
- `__init__.py`: `RAGEvaluator` import/export 제거
- `runner.py`: `RAGEvaluator` import 제거, `isinstance(result, EvaluationReport)` 양방향 호환 분기 제거 → v0.4 `EvaluationReport` 전용
- `run_all_combos.py`: `--legacy-evaluator` CLI 인자 제거, 두 곳의 v0.3 분기 (`args.legacy_evaluator`) 제거

### Phase 3: parent_store 포맷 통일
- `graph/nodes.py`의 `retrieve_parent_chunks()`:
  - 기존: 개별 `{parent_id}.json` 파일 로드
  - 변경: `parents.json` 단일 딕셔너리에서 `parent_id` 키 조회
  - 캐싱: 클로저 내 `_parents_cache` dict로 한 번만 로드
  - `chunker.py`의 출력 포맷(`parents.json`)과 일치

### Phase 4: pyproject.toml 중복 제거
- `rag_bench/pyproject.toml` 삭제 (35줄)
- Root `pyproject.toml`이 상위 집합으로 이미 존재

### 검증 결과
- `python3 -m py_compile`: 11개 파일 모두 통과
- import 테스트: `dotenv` 미설치로 런타임 import 불가 (기존 환경 문제, 리팩토링과 무관)
- 72개 full 프리셋 벤치마크: 72개 전략 성공 / 0개 실패

### 잔여 작업
- `rag_bench_colab/rag_benchmark.ipynb`: 셀 순서 변경 미커밋 (리팩토링 범위 외)
- `ARCHITECTURE.md`, `STACK.md`: untracked 상태
