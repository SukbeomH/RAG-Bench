# Current Session Context

## Session Narrative
> On 2026-03-04, implemented `rag-pipeline` package — LangGraph StateGraph 기반 파이프라인 재구성. 3개 그래프(RAG, RAG Bench, PDF Bench), 7개 노드, FastAPI v2 + CLI. 22 tests all passed.

## Context Snapshot
- **Active Task**: feat: LangGraph 기반 rag-pipeline 패키지 신규 생성
- **Branch**: master
- **Last Updated**: 2026-03-04

## Key Files
```
packages/rag-pipeline/src/autorag_pipeline/
  states/rag_state.py          — RAGState TypedDict
  nodes/{parse,index,retrieve,generate}.py — RAG 노드
  nodes/{bench_prep,bench_run,bench_eval,pdf_bench}.py — 벤치마크 노드
  graphs/{rag_pipeline,rag_bench_graph,pdf_bench_graph}.py — 3개 그래프
  integration/{api_adapter,cli}.py — API v2 + CLI
```

## Test Results
- rag-pipeline: 22/22 passed
- 전체: 227 passed (기존 10 fixture errors 무관)

## Next Steps
- E2E 통합 테스트 (실제 PDF + LLM)
- Agentic RAG 확장
- K8s 오케스트레이터 → LangGraph 교체
