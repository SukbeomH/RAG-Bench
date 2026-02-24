"""
Graph Builder 모듈.

전략(strategy) 객체를 주입받아 LangGraph 에이전트를 구성/컴파일한다.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_bench.base import BaseRAGStrategy
from rag_bench.config import (
    DEFAULT_AGENT_LLM,
    DEFAULT_LLM_TEMPERATURE,
    PARENT_STORE_PATH,
    make_llm,
)
from rag_bench.graph.nodes import (
    create_tools,
    extract_final_answer,
    human_input_node,
    make_agent_node,
    make_aggregate_responses,
    make_analyze_and_rewrite_query,
    make_analyze_chat_and_summarize,
    route_after_rewrite,
)
from rag_bench.graph.state import AgentState, State


def build_agent_graph(
    strategy: BaseRAGStrategy,
    llm_model: str = DEFAULT_AGENT_LLM,
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE,
    parent_store_path: str = str(PARENT_STORE_PATH),
):
    """
    전략 기반 LangGraph 에이전트를 구성하고 컴파일한다.

    Args:
        strategy: 사용할 RAG 전략.
        llm_model: LLM 모델명.
        llm_temperature: LLM 온도.
        parent_store_path: Parent chunk JSON 저장 경로.

    Returns:
        compiled graph (agent_graph).
    """
    # LLM 초기화
    llm = make_llm(llm_model, temperature=llm_temperature)

    # 전략 기반 도구 생성
    tools = create_tools(strategy, parent_store_path)
    llm_with_tools = llm.bind_tools(tools)

    # 체크포인터
    checkpointer = InMemorySaver()

    # Agent subgraph
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("agent", make_agent_node(llm_with_tools))
    agent_builder.add_node("tools", ToolNode(tools))
    agent_builder.add_node("extract_answer", extract_final_answer)

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "extract_answer"}
    )
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("extract_answer", END)
    agent_subgraph = agent_builder.compile()

    # Main graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", make_analyze_chat_and_summarize(llm))
    graph_builder.add_node("analyze_rewrite", make_analyze_and_rewrite_query(llm))
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("process_question", agent_subgraph)
    graph_builder.add_node("aggregate", make_aggregate_responses(llm))

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_edge(["process_question"], "aggregate")
    graph_builder.add_edge("aggregate", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"],
    )

    print(f"Agent graph compiled with strategy: {strategy.name}")
    return agent_graph
