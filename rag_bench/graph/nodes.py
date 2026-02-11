"""
Graph 노드 및 라우팅 로직 모듈.

LangGraph 에이전트의 각 노드(요약, 분석, 검색, 응답 통합)와
라우팅 로직을 정의한다.

전략(strategy) 객체를 주입받아 검색 도구를 동적으로 바인딩한다.
"""

import json
import os
from typing import Literal, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.types import Send

from rag_bench.graph.prompts import (
    get_aggregation_prompt,
    get_conversation_summary_prompt,
    get_query_analysis_prompt,
    get_rag_agent_prompt,
)
from rag_bench.graph.state import AgentState, QueryAnalysis, State


def create_tools(strategy, parent_store_path: str):
    """
    전략 객체와 parent store 경로를 기반으로 LangGraph 도구를 생성한다.

    Args:
        strategy: BaseRAGStrategy 구현체.
        parent_store_path: Parent 청크 JSON 저장 경로.

    Returns:
        (tools, llm_with_tools): 도구 목록, 도구가 바인딩된 LLM.
    """

    @tool
    def search_child_chunks(query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            results = strategy.retrieve(query, k=limit)
            if not results:
                return "NO_RELEVANT_CHUNKS"
            return "\n\n".join(
                [
                    f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                    f"File Name: {doc.metadata.get('source', '')}\n"
                    f"Content: {doc.page_content.strip()}"
                    for doc in results
                ]
            )
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.

        Args:
            parent_id: Parent chunk ID to retrieve
        """
        file_name = (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        path = os.path.join(parent_store_path, file_name)
        if not os.path.exists(path):
            return "NO_PARENT_DOCUMENT"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            f"Parent ID: {parent_id}\n"
            f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {data.get('page_content', '').strip()}"
        )

    return [search_child_chunks, retrieve_parent_chunks]


# ---------------------------------------------------------------------------
# 그래프 노드 함수들
# ---------------------------------------------------------------------------


def make_analyze_chat_and_summarize(llm):
    """대화 요약 노드 팩토리."""

    def analyze_chat_and_summarize(state: State):
        if len(state["messages"]) < 4:
            return {"conversation_summary": ""}
        relevant_msgs = [
            msg
            for msg in state["messages"][:-1]
            if isinstance(msg, (HumanMessage, AIMessage))
            and not getattr(msg, "tool_calls", None)
        ]
        if not relevant_msgs:
            return {"conversation_summary": ""}
        conversation = "Conversation history:\n"
        for msg in relevant_msgs[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation += f"{role}: {msg.content}\n"
        summary_response = llm.with_config(temperature=0.2).invoke(
            [SystemMessage(content=get_conversation_summary_prompt())]
            + [HumanMessage(content=conversation)]
        )
        return {
            "conversation_summary": summary_response.content,
            "agent_answers": [{"__reset__": True}],
        }

    return analyze_chat_and_summarize


def make_analyze_and_rewrite_query(llm):
    """쿼리 분석/재작성 노드 팩토리."""

    def analyze_and_rewrite_query(state: State):
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")
        context_section = (
            f"Conversation Context:\n{conversation_summary}\n"
            if conversation_summary.strip()
            else ""
        ) + f"User Query:\n{last_message.content}\n"

        llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(
            QueryAnalysis
        )
        response = llm_with_structure.invoke(
            [SystemMessage(content=get_query_analysis_prompt())]
            + [HumanMessage(content=context_section)]
        )

        if len(response.questions) > 0 and response.is_clear:
            delete_all = [
                RemoveMessage(id=m.id)  # type: ignore[arg-type]
                for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            return {
                "questionIsClear": True,
                "messages": delete_all,
                "originalQuery": last_message.content,
                "rewrittenQuestions": response.questions,
            }
        else:
            clarification = (
                response.clarification_needed
                if response.clarification_needed
                and len(response.clarification_needed.strip()) > 10
                else "I need more information to understand your question."
            )
            return {
                "questionIsClear": False,
                "messages": [AIMessage(content=clarification)],
            }

    return analyze_and_rewrite_query


def human_input_node(state: State):
    """사용자 입력 대기 노드."""
    return {}


def route_after_rewrite(
    state: State,
) -> Union[Literal["human_input", "process_question"], list[Send]]:
    """쿼리 분석 후 라우팅."""
    if not state.get("questionIsClear", False):
        return "human_input"
    return [
        Send(
            "process_question",
            {"question": query, "question_index": idx, "messages": []},
        )
        for idx, query in enumerate(state["rewrittenQuestions"])
    ]


def make_agent_node(llm_with_tools):
    """RAG 에이전트 노드 팩토리."""

    def agent_node(state: AgentState):
        sys_msg = SystemMessage(content=get_rag_agent_prompt())
        if not state.get("messages"):
            human_msg = HumanMessage(content=state["question"])
            response = llm_with_tools.invoke([sys_msg] + [human_msg])
            return {"messages": [human_msg, response]}
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    return agent_node


def extract_final_answer(state: AgentState):
    """에이전트의 최종 답변 추출."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {
                "final_answer": msg.content,
                "agent_answers": [
                    {
                        "index": state["question_index"],
                        "question": state["question"],
                        "answer": msg.content,
                    }
                ],
            }
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": "Unable to generate an answer.",
            }
        ],
    }


def make_aggregate_responses(llm):
    """응답 통합 노드 팩토리."""

    def aggregate_responses(state: State):
        if not state.get("agent_answers"):
            return {"messages": [AIMessage(content="No answers were generated.")]}
        sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])
        formatted_answers = ""
        for i, ans in enumerate(sorted_answers, start=1):
            formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"
        user_message = HumanMessage(
            content=f"Original user question: {state['originalQuery']}\n"
            f"Retrieved answers:{formatted_answers}"
        )
        synthesis_response = llm.invoke(
            [SystemMessage(content=get_aggregation_prompt())] + [user_message]
        )
        return {"messages": [AIMessage(content=synthesis_response.content)]}

    return aggregate_responses
