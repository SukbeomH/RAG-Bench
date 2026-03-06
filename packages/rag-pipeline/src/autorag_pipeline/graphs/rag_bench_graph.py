"""RAG 벤치마크 파이프라인 — LangGraph StateGraph.

HuggingFace 데이터셋 로드 → Contextual 보강 → 전략 실행 → RAGAS 평가.
5 Dense × 2 Sparse × 2 Reranker = 20개 조합을 순회하며 RAGAS 메트릭을 산출.

노드 역할:
  - load_hf_data: HuggingFace에서 QA 데이터셋 로드 (카테고리별)
  - enrich_contextual: QA 쌍에 Contextual 메타데이터 보강
  - run_benchmark: BenchmarkRunner로 전략 조합별 검색+답변 실행
  - evaluate_ragas: RAGAS 4종 메트릭(Context Precision/Recall, Faithfulness, Relevancy) 평가

사용 예::

    from autorag_pipeline.graphs.rag_bench_graph import build_rag_bench_graph

    graph = build_rag_bench_graph()
    result = graph.invoke({
        "category": "LEGAL",
        "combo_specs": [...],
    })
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.bench_eval import evaluate_ragas
from autorag_pipeline.nodes.bench_prep import enrich_contextual, load_hf_data
from autorag_pipeline.nodes.bench_run import run_benchmark
from autorag_pipeline.states.rag_bench_state import RAGBenchState


def _should_prep(state: RAGBenchState) -> str:
    """데이터 준비 완료 여부에 따라 분기. prep_done=True면 데이터 로드 건너뜀."""
    if state.get("prep_done"):
        return "run_benchmark"
    return "load_hf_data"


def build_rag_bench_graph() -> StateGraph:
    """RAG 벤치마크 파이프라인 그래프를 빌드+컴파일.

    Graph flow::

        START → [prep_done?]
          ├─ False → load_hf_data → enrich_contextual → run_benchmark → evaluate_ragas → END
          └─ True  → run_benchmark → evaluate_ragas → END
    """
    graph = StateGraph(RAGBenchState)

    graph.add_node("load_hf_data", load_hf_data)
    graph.add_node("enrich_contextual", enrich_contextual)
    graph.add_node("run_benchmark", run_benchmark)
    graph.add_node("evaluate_ragas", evaluate_ragas)

    graph.add_conditional_edges(
        START,
        _should_prep,
        {"load_hf_data": "load_hf_data", "run_benchmark": "run_benchmark"},
    )
    graph.add_edge("load_hf_data", "enrich_contextual")
    graph.add_edge("enrich_contextual", "run_benchmark")
    graph.add_edge("run_benchmark", "evaluate_ragas")
    graph.add_edge("evaluate_ragas", END)

    return graph.compile()
