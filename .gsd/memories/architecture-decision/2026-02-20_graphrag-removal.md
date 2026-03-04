---
title: "GraphRAG 전략 완전 제거"
tags:
  - architecture-decision
  - graphrag
  - lightrag
  - removal
  - dependency
type: architecture-decision
created: 2026-02-20T00:00:00+09:00
contextual_description: "LightRAG 의존성 문제 및 실용성 부족으로 GraphRAGStrategy를 모든 코드베이스에서 완전 제거. 총 콤보 수 73 → 72개."
keywords:
  - GraphRAG
  - LightRAG
  - lightrag-hku
  - 의존성 제거
  - ComboSpec
  - for_graphrag
related:
  - 2026-02-11_graphrag-strategy-implementation
  - 2026-02-20_benchmark-efficiency-api-extension-html-report
---

## GraphRAG 전략 완전 제거

### 결정 배경
- **LightRAG 의존성 문제**: `lightrag-hku` 패키지가 설치/호환성 이슈 반복
- **실용성 부족**: GraphRAG는 구성 복잡도 대비 벤치마크 환경에서 성능 이점 미미
- **유지보수 부담**: 별도 그래프 생성 파이프라인, 비동기 처리 (`nest-asyncio`) 필요

### 결정
GraphRAGStrategy를 코드베이스에서 완전 제거. 이는 되돌리기 어려운 결정이므로 신중하게 실행.

---

### 제거 범위

#### 삭제된 파일
- `rag_bench/strategies/graph_rag.py` (GraphRAGStrategy 전체)

#### 수정된 파일 (GraphRAG 관련 코드 제거)

**`rag_bench/strategies/__init__.py`**
- `GraphRAGStrategy` import 및 `__all__` 항목 제거

**`rag_bench/scripts/run_all_combos.py`**
- `ComboSpec` 데이터클래스: `graphrag: bool = False` 필드 제거
- `ComboSpec.for_graphrag()` classmethod 제거
- `ComboSpec.label`, `retrieval_mode`, `index_key` 프로퍼티에서 graphrag 분기 제거
- `generate_valid_combinations(config)`: `include_graphrag` 파라미터 제거
- `build_strategy_from_spec()`: GraphRAG 분기 제거
- `_try_build_graphrag()` 함수 전체 제거
- `--skip_graphrag` argparse 옵션 제거
- docstring: "총 유효 조합: 4 × 3 × 6 = 72개"로 업데이트

**`rag_bench_colab/colab_runner.py`**
- `ColabBenchmarkRunner.__init__()`: `include_graphrag` 파라미터 제거
- `generate_combos()`: `include_graphrag` 전달 제거
- `run_graphrag()` 메서드 전체 제거
- GraphRAG 체크포인트 로직 제거
- `export_results()`: `graphrag_result` 파라미터 제거
- `_generate_report()`: `graphrag_result` 파라미터 및 GraphRAG 결과 섹션 제거

**`rag_bench_colab/rag_benchmark.ipynb`**
- cell-0 (markdown): GraphRAG 행 및 "+GraphRAG" 참조 제거
- cell-6 (Section 2 markdown): `INCLUDE_GRAPHRAG` 옵션 설명 제거
- cell-7 (Section 2 settings): `INCLUDE_GRAPHRAG = True` 변수 제거
- cell-9 (Section 3 runner): `include_graphrag=INCLUDE_GRAPHRAG` 인수 제거
- cell-19 (Section 7): GraphRAG 섹션 헤더 → 시각화 대시보드 헤더로 변환
- cell-20, cell-21: GraphRAG 실행 셀 삭제
- cell-29 (비용 요약): `est_graphrag_indexing_cost` 및 GraphRAG 비용 항목 제거

**`pyproject.toml`**
- `"lightrag-hku>=1.0"` 제거
- `"nest-asyncio>=1.6"` 제거

---

### 영향 분석
- **총 콤보 수**: 73개 → 72개
- **하위 호환성**: `generate_valid_combinations()` 시그니처 변경 — `include_graphrag` 키워드 인수 제거됨
- **외부 사용자 영향**: `ColabBenchmarkRunner(include_graphrag=...)` 패턴 사용 불가
- **의존성**: `lightrag-hku`, `nest-asyncio` 더 이상 필요 없음

---

### 재구현 시 참고
`2026-02-11_graphrag-strategy-implementation.md` 에 GraphRAGStrategy 구현 상세가 기록되어 있음.
LightRAG API: `AsyncLightRAG`, `LightRAGParam`, `QueryParam` 사용. 재도입 필요 시 해당 메모리 참조.
