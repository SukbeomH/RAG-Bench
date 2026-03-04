"""Generate node: LLM answer generation with citations."""

from __future__ import annotations

from typing import Any

from autorag_pipeline.states.rag_state import RAGState

_SYSTEM_PROMPT = (
    "당신은 문서 기반 질의응답 어시스턴트입니다. "
    "제공된 컨텍스트만을 사용하여 질문에 답변하세요. "
    "답변에 사용한 출처를 [번호] 형식으로 인용하세요. "
    "컨텍스트에 답이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다.'라고 답하세요."
)


def generate_answer(state: RAGState) -> dict[str, Any]:
    """Generate an answer using the LLM with retrieved context."""
    from autorag_retrieval.config import make_llm, DEFAULT_ANSWER_LLM

    query: str = state["query"]
    context: str = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs", [])

    llm = make_llm(model=DEFAULT_ANSWER_LLM, temperature=0)

    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"컨텍스트:\n{context}\n\n질문: {query}"),
    ]
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    citations: list[dict] = []
    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        citations.append(
            {
                "chunk_id": meta.get("chunk_id", ""),
                "source_path": meta.get("source_path", ""),
                "page_number": meta.get("page_number", 0),
                "text_snippet": doc.page_content[:200],
                "bbox": meta.get("bbox"),
                "relevance_score": 0.0,
            }
        )

    return {"answer": answer, "citations": citations}
