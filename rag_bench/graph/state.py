"""
State 정의 모듈.

LangGraph 에이전트의 메인 State 및 서브 State를 정의한다.
"""

from typing import Annotated, List

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """
    에이전트 답변을 누적하거나 초기화하는 리듀서.
    __reset__ 플래그가 있으면 목록을 초기화한다.
    """
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


class State(MessagesState):
    """메인 그래프 State."""

    questionIsClear: bool = False  # type: ignore[misc]
    conversation_summary: str = ""  # type: ignore[misc]
    originalQuery: str = ""  # type: ignore[misc]
    rewrittenQuestions: List[str] = []  # type: ignore[misc]
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []  # type: ignore[misc]


class AgentState(MessagesState):
    """에이전트 서브그래프 State."""

    question: str = ""  # type: ignore[misc]
    question_index: int = 0  # type: ignore[misc]
    final_answer: str = ""  # type: ignore[misc]
    agent_answers: List[dict] = []  # type: ignore[misc]


class QueryAnalysis(BaseModel):
    """쿼리 분석 결과 (Structured Output)."""

    is_clear: bool = Field(description="사용자의 질문이 명확하고 답변 가능한지 여부.")
    questions: List[str] = Field(description="재작성된 자기완결적 질문 목록.")
    clarification_needed: str = Field(description="질문이 불명확한 경우 설명.")
