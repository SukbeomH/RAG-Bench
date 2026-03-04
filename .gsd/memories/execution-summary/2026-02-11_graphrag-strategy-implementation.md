---
title: "GraphRAGStrategy 구현 (LightRAG 기반)"
tags:
  - execution
  - summary
  - graphrag
  - lightrag
  - knowledge-graph
  - retrieval
  - strategy
type: execution-summary
created: 2026-02-11T21:00:00+09:00
contextual_description: "LightRAG 백엔드로 GraphRAGStrategy 전체 구현 완료. 엔티티-관계 지식 그래프 구축 + hybrid 검색 모드, gpt-4.1-nano 기본 LLM."
keywords:
  - GraphRAG
  - LightRAG
  - knowledge-graph
  - entity-extraction
  - relationship-extraction
  - hybrid-search
  - gpt-4.1-nano
  - NetworkX
  - NanoVectorDB
  - async-wrapping
  - nest_asyncio
related:
  - 2026-02-11_colbert-strategy-implementation
  - 2026-02-11_project-memory-sync
---

## GraphRAGStrategy 구현 (LightRAG 기반)

### 실행 내용
스텁 상태였던 `rag_bench/strategies/graph_rag.py`를 LightRAG 백엔드로 전체 구현.

### 변경 파일
| 파일 | 변경 |
|------|------|
| `rag_bench/strategies/graph_rag.py` | 스텁 → 전체 구현 (235줄) |
| `pyproject.toml` | `lightrag-hku>=1.0`, `nest-asyncio>=1.6` 의존성 추가 |
| `.gitignore` | `lightrag_index/` 추가 |

### 구현 상세
- **GraphRAGRetriever** (~10 LOC): `BaseRetriever` 상속, `strategy.retrieve()` 위임, `ConfigDict(arbitrary_types_allowed=True)`
- **GraphRAGStrategy** (~180 LOC):
  - 기본 LLM: `gpt-4.1-nano` (GPT-4o-mini 대비 입력 7.5배, 출력 4배 저렴)
  - `llm_model` 파라미터로 임의 OpenAI 모델 지정 가능
  - **`_run_async(coro)`**: async→sync 래핑 헬퍼
    - `asyncio.get_running_loop()` 존재 시 (Jupyter 등) → `nest_asyncio.apply()` 적용
    - 없으면 `asyncio.run(coro)` 사용
  - **`_ensure_initialized()`**: LightRAG 인스턴스 lazy 생성
    - `openai_complete_if_cache`로 커스텀 LLM 함수 생성 (모델명 파라미터화)
    - `openai_embed` 임베딩 함수 사용
    - 스토리지: JsonKV + NanoVectorDB + NetworkX + JsonDocStatus (파일 기반, 서버 불필요)
    - `initialize_storages()` + `initialize_pipeline_status()` 호출
  - **`index(documents)`**: `ainsert(texts)` → LLM 기반 엔티티/관계 추출 → 그래프 구축
  - **`retrieve(query, k)`**: `aquery(only_need_context=True, mode=hybrid)` → 컨텍스트 파싱 → `List[Document]`
  - **`cleanup()`**: `finalize_storages()` + `shutil.rmtree(working_dir)`
  - 검색 모드 지원: local, global, hybrid, naive, mix

### 설계 결정
1. **gpt-4.1-nano 기본 선택**: 그래프 구축 시 대량 LLM 호출 발생 → 비용 효율 우선
   - GPT-4.1-nano: $0.02/$0.15 per 1M tokens
   - GPT-4o-mini: $0.15/$0.60 per 1M tokens
2. **커스텀 LLM 함수**: `openai_complete_if_cache`로 래핑하여 `llm_model` 파라미터에 아무 OpenAI 모델명 전달 가능
3. **nest_asyncio**: Jupyter 환경에서 이벤트 루프 중첩 문제 해결

### 검증 결과
- import 검증 통과: `from rag_bench.strategies import GraphRAGStrategy`
- 클래스 생성, name/description/is_ready 프로퍼티 정상 동작
- 모든 모드(local, global, hybrid, naive) 인스턴스 생성 확인
- **주의**: `index()`/`retrieve()` E2E 테스트는 OpenAI API 키 필요 (비용 발생)

### 커밋
```
da5bced feat: GraphRAGStrategy 구현 — LightRAG 기반 지식 그래프 RAG 전략
```

### 현재 전략 구현 상태
| 전략 | 상태 | 비고 |
|------|------|------|
| `DenseSparseStrategy` | **완료** | 6가지 임베딩 조합 (Qdrant 하이브리드) |
| `ColBERTStrategy` | **완료** | PyLate 기반, brute-force + Voyager |
| `ColBERTRerankStrategy` | **완료** | 2단계 리랭킹 (임의 1차 전략 + ColBERT MaxSim) |
| `GraphRAGStrategy` | **완료** | LightRAG 기반, gpt-4.1-nano, hybrid 모드 |
