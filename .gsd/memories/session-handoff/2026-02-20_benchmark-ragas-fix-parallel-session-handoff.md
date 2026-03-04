---
title: "세션 인수인계: 벤치마크 실행 + RAGAS 수정 + 병렬화 구현"
type: session-handoff
date: 2026-02-20
tags: [benchmark, ragas, parallel, bug-fix, memory-oom]
---

## 세션 요약

standard 프리셋 벤치마크 실행, RAGAS 평가 오류 수정, Pass 1 병렬화 구현, OOM 분석 문서화.

---

## 완료된 작업

### 1. Pydantic 오류 수정 (`upstage_embed.py`)
- `class _UpstageRetriever` 내부의 `from pydantic import ConfigDict` → 파일 최상단으로 이동
- 수정 파일: `rag_bench/strategies/upstage_embed.py`

### 2. RAGAS 평가 오류 수정 (`evaluator.py`)

**문제 1**: `gpt-4o-nano` → 존재하지 않는 모델명
- 수정: `llm_model: str = "gpt-4o-mini"` (line 284)

**문제 2**: `InstructorLLM.generate() got an unexpected keyword argument 'n'`
- 원인: `llm_factory()` 반환 타입이 `InstructorLLM` (RAGAS `BaseRagasLLM` 아님) → `agenerate_text()` 미지원
- 수정: `_ensure_initialized()`에서 `llm_factory()` → `LangchainLLMWrapper(ChatOpenAI(...))` 교체
- `_MultiPerspectiveLLM.generate()` async 메서드 추가 (n 파라미터 지원)
- 수정 파일: `rag_bench/evaluation/evaluator.py`

```python
# 변경 전
from ragas.llms import llm_factory
base_llm = llm_factory(model=self.llm_model, client=openai_client)

# 변경 후
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
chat_llm = ChatOpenAI(model=self.llm_model, http_client=..., http_async_client=...)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    base_llm = LangchainLLMWrapper(chat_llm)
```

### 3. Pass 1 병렬화 구현

**`rag_bench/runner.py`**:
- `BenchmarkRunner.__init__`에 `parallel_strategies: int = 0` 파라미터 추가
- `_run_strategy_all_queries()` 메서드 추가 (단일 전략 전체 쿼리 실행)
- `_run_strategies_parallel()` 메서드 추가 (ThreadPoolExecutor 기반)
- `run()` 진입부에 `parallel_strategies > 1` 분기 추가

**`rag_bench/scripts/run_all_combos.py`**:
- `--pass1-workers N` CLI 옵션 추가
- `BenchmarkRunner` 생성 시 `parallel_strategies=pass1_workers` 전달

**사용법**:
```bash
python -m rag_bench.scripts.run_all_combos --preset standard --pass1-workers 4
```

### 4. `KoreanBM25Encoder` Thread-Safety 수정 (`dense_sparse.py`)
- `threading.Lock` import 추가
- `__init__`에 `self._vocab_lock = threading.Lock()` 추가
- `_get_or_create_id()`를 `with self._vocab_lock:` 블록으로 감쌈
- 병렬 실행 시 vocab 동시 쓰기 race condition 방지

### 5. OOM 분석 문서화
- `docs/memory_oom_analysis.md` 신규 작성
- 컴포넌트별 디바이스/메모리, 위험 요소, 개선 과제 정리

---

## 벤치마크 실행 결과 (standard 프리셋, 1차 — RAGAS 수정 전)

- 총 소요: 2193.5초, 전략 24개 성공
- **RAGAS 점수 모두 0.0000** (InstructorLLM 오류로 실패)
- 레이턴시 Best: `DS(all-MiniLM-L6-v2+fastembed_bm25)` **43ms**
- 레이턴시 Worst: FlashRank Rerank + multilingual-e5-large **~1900ms**

## 2차 실행 (RAGAS 수정 후) — 미완료

- RAGAS 수정 후 재실행 시작 (인덱싱 재실행 중 user 종료)
- RAGAS 수정이 실제로 동작하는지 검증 필요

---

## 미완료 작업 / 다음 세션

1. **RAGAS 수정 검증**: `--preset standard` 재실행 후 weighted 점수 확인
2. **병렬 실행 검증**: `--pass1-workers 4` 실제 동작 + 속도 비교
3. **SPLADE 싱글톤**: `IndexCacheManager`에 `_splade_cache` 추가 (OOM 문서의 High 우선순위 항목)
4. **인덱스 재사용 문제**: `--preset standard` 재실행 시에도 재인덱싱 발생 (원인 미확인)

---

## 파일 변경 목록

| 파일 | 변경 |
|------|------|
| `rag_bench/strategies/upstage_embed.py` | Pydantic ConfigDict 임포트 위치 수정 |
| `rag_bench/evaluation/evaluator.py` | gpt-4o-mini 수정, LangchainLLMWrapper 교체, generate() 추가 |
| `rag_bench/runner.py` | parallel_strategies 파라미터 + 병렬 실행 메서드 추가 |
| `rag_bench/strategies/dense_sparse.py` | threading.Lock으로 vocab thread-safety 보장 |
| `rag_bench/scripts/run_all_combos.py` | --pass1-workers CLI 옵션 + BenchmarkRunner 연결 |
| `docs/memory_oom_analysis.md` | OOM 분석 문서 신규 작성 |
