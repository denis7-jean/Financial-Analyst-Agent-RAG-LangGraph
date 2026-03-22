from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage

from src.core.state import AgentState
from src.core.memory import get_checkpointer, get_store
from src.nodes.router import router_node
from src.nodes.filing import (
    filing_financial_node, filing_narrative_node, cell_plan_node
)
from src.nodes.market_news import market_node, news_node
from src.nodes.math_engine import math_prep_node, calculator_node
from src.nodes.generator import draft_answer_node
from src.nodes.auditor import (
    financial_audit_node, narrative_audit_node, revision_plan_node
)
from src.edges.conditions import (
    route_after_router, requires_math, check_cell_confidence,
    check_math_confidence, route_to_audit, check_audit, route_revision
)

workflow = StateGraph(AgentState)

# --- nodes ---
workflow.add_node("router_node", router_node)
workflow.add_node("filing_financial_node", filing_financial_node)
workflow.add_node("cell_plan_node", cell_plan_node)
workflow.add_node("filing_narrative_node", filing_narrative_node)
workflow.add_node("market_node", market_node)
workflow.add_node("news_node", news_node)
workflow.add_node("math_prep_node", math_prep_node)
workflow.add_node("calculator_node", calculator_node)
workflow.add_node("draft_answer_node", draft_answer_node)
workflow.add_node("financial_audit_node", financial_audit_node)
workflow.add_node("narrative_audit_node", narrative_audit_node)
workflow.add_node("revision_plan_node", revision_plan_node)

# --- edges ---
workflow.add_edge(START, "router_node")
workflow.add_conditional_edges(
    "router_node",
    route_after_router,
    {
        "filing_financial": "filing_financial_node",
        "filing_narrative": "filing_narrative_node",
        "market": "market_node",
        "news": "news_node",
        "general": "draft_answer_node",
    },
)
workflow.add_conditional_edges(
    "filing_financial_node",
    requires_math,
    {
        "cell_plan_node": "cell_plan_node",
        "draft_answer_node": "draft_answer_node",
    },
)
workflow.add_conditional_edges(
    "cell_plan_node",
    check_cell_confidence,
    {
        "math_prep_node": "math_prep_node",
        "draft_answer_node": "draft_answer_node",
    },
)
workflow.add_edge("filing_narrative_node", "draft_answer_node")
workflow.add_conditional_edges(
    "math_prep_node",
    check_math_confidence,
    {
        "calculator_node": "calculator_node",
        "draft_answer_node": "draft_answer_node",
    },
)
workflow.add_edge("calculator_node", "draft_answer_node")
workflow.add_edge("market_node", "draft_answer_node")
workflow.add_edge("news_node", "draft_answer_node")
workflow.add_conditional_edges(
    "draft_answer_node",
    route_to_audit,
    {
        "financial_audit_node": "financial_audit_node",
        "narrative_audit_node": "narrative_audit_node",
    },
)
workflow.add_conditional_edges(
    "financial_audit_node",
    check_audit,
    {
        "END": END,
        "revision_plan_node": "revision_plan_node",
    },
)
workflow.add_conditional_edges(
    "narrative_audit_node",
    check_audit,
    {
        "END": END,
        "revision_plan_node": "revision_plan_node",
    },
)
workflow.add_conditional_edges(
    "revision_plan_node",
    route_revision,
    {
        "cell_plan_node": "cell_plan_node",
        "draft_answer_node": "draft_answer_node",
        "filing_financial_node": "filing_financial_node",
    },
)

checkpointer = get_checkpointer()
store = get_store()
app = workflow.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    query = "Compare Apple's 2023 and 2024 net sales and tell me how much higher 2024 was."
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "graph-math-smoke-test"}},
    )
    final_message = result["messages"][-1]
    print(getattr(final_message, "content", final_message))
