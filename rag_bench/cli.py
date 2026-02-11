"""
CLI 인터페이스 모듈.

Jupyter 노트북 또는 스크립트에서 RAG 에이전트와 대화하는 인터페이스.
"""

import uuid

from langchain_core.messages import HumanMessage


class RAGChat:
    """
    RAG 에이전트 대화 인터페이스.

    Usage:
        from rag_bench.strategies import DenseSparseStrategy
        from rag_bench.graph import build_agent_graph
        from rag_bench.cli import RAGChat

        strategy = DenseSparseStrategy(combo_id=1)
        graph = build_agent_graph(strategy)
        chat = RAGChat(graph, strategy)

        chat.ask("쿠버네티스 Pod란 무엇인가요?")
        chat.clear()
    """

    def __init__(self, agent_graph, strategy):
        self._graph = agent_graph
        self._strategy = strategy
        self._config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def ask(self, query: str) -> str:
        """
        질문을 전송하고 응답을 반환한다.

        Args:
            query: 사용자 질문.

        Returns:
            에이전트 응답 텍스트.
        """
        if not query.strip():
            print("유효한 질문을 입력하세요.")
            return ""

        print(f"Query: {query}")
        print(f"Strategy: {self._strategy.name}")
        print("Thinking...\n")

        current_state = self._graph.get_state(self._config)
        if current_state.next:
            self._graph.update_state(
                self._config,
                {"messages": [HumanMessage(content=query.strip())]},
            )
            result = self._graph.invoke(None, self._config)
        else:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=query.strip())]},
                self._config,
            )

        response = result["messages"][-1].content
        print("Assistant:")
        print(response)
        return response

    def clear(self) -> None:
        """현재 세션을 초기화하고 새 대화를 시작한다."""
        try:
            self._graph.checkpointer.delete_thread(
                self._config["configurable"]["thread_id"]
            )
        except Exception:
            pass
        self._config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        print("Session cleared. New conversation started.")

    def ask_display(self, query: str) -> None:
        """
        Jupyter 노트북용: Markdown으로 응답을 렌더링한다.

        Args:
            query: 사용자 질문.
        """
        response = self.ask(query)
        try:
            from IPython.display import Markdown, display

            display(Markdown(response))
        except ImportError:
            pass
