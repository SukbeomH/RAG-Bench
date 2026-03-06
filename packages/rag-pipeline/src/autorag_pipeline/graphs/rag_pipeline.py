"""기본 RAG 파이프라인 — LangGraph StateGraph.

PDF 문서를 파싱 → 청킹 → 인덱싱 → 검색 → 답변 생성하는 E2E 파이프라인.

노드 역할:
  - parse_pdf: autorag_parsers로 PDF→마크다운 변환
  - chunk_document: 마크다운을 검색 단위 청크로 분할
  - build_and_index: 청크를 Qdrant에 임베딩+인덱싱 (최초 1회)
  - retrieve: 사용자 질의로 관련 청크 검색 (Dense+Sparse 하이브리드)
  - generate_answer: 검색된 Context + 질의를 LLM에 전달하여 응답 생성

사용 예::

    from autorag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()
    result = graph.invoke({
        "pdf_path": "document.pdf",
        "query": "KDB 아키텍처 설계 원칙은?",
    })
    print(result["answer"])
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.generate import generate_answer
from autorag_pipeline.nodes.index import build_and_index
from autorag_pipeline.nodes.parse import chunk_document, parse_pdf
from autorag_pipeline.nodes.retrieve import retrieve
from autorag_pipeline.states.rag_state import RAGState


def _should_index(state: RAGState) -> str:
    """인덱스 존재 여부에 따라 분기. index_ready=True면 인덱싱 건너뜀."""
    if state.get("index_ready"):
        return "retrieve"
    return "build_and_index"


def build_rag_pipeline() -> StateGraph:
    """기본 RAG 파이프라인 그래프를 빌드+컴파일.

    Graph flow::

        START → parse_pdf → chunk_document → [index_ready?]
          ├─ False → build_and_index → retrieve → generate_answer → END
          └─ True  → retrieve → generate_answer → END
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("parse_pdf", parse_pdf)
    graph.add_node("chunk_document", chunk_document)
    graph.add_node("build_and_index", build_and_index)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)

    # Edges
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "chunk_document")
    graph.add_conditional_edges(
        "chunk_document",
        _should_index,
        {"build_and_index": "build_and_index", "retrieve": "retrieve"},
    )
    graph.add_edge("build_and_index", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
