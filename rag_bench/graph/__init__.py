"""graph 서브패키지 — LangGraph 통합.

LangGraph 의존성이 설치되지 않은 환경에서도
패키지의 다른 부분은 정상적으로 사용할 수 있도록 lazy import를 적용한다.
"""


def build_agent_graph(*args, **kwargs):
    """전략 기반 LangGraph 에이전트를 구성하고 컴파일한다 (lazy import)."""
    from rag_bench.graph.builder import build_agent_graph as _build

    return _build(*args, **kwargs)


__all__ = ["build_agent_graph"]
