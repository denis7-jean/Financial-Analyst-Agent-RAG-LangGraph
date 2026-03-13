from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
from src.config import GEMINI_MODEL, GOOGLE_API_KEY
from src.tools.tools import calculator, search_10k, web_search, yfinance_tool

TOOLS = [search_10k, yfinance_tool, calculator, web_search]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
).bind_tools(TOOLS)


def agent(state: AgentState) -> AgentState:
    """Run the analyst LLM with the current conversation state."""
    system = SystemMessage(
        content=(
            "You are a professional financial analyst.\n"
            "\n"
            "Tool usage rules:\n"
            "1. Use search_10k for answers grounded in the ingested filings.\n"
            "2. Use yfinance_tool for market data such as price, P/E, market cap, or trading context.\n"
            "3. Use web_search for macroeconomic or news questions that are not contained in the filings.\n"
            "4. If a question involves arithmetic, growth rates, differences, margins, or projections, "
            "you must call calculator.\n"
            "5. Do not perform mental math.\n"
            "6. Cite filing evidence with source, page, and section when using retrieved filing context.\n"
        )
    )
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


tools_node = ToolNode(TOOLS)


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


if __name__ == "__main__":
    query = (
        "What was Apple's total net sales in 2024? "
        "If I assume a 5% increase for next year, what would that be?"
    )

    print("Streaming agent execution:\n")
    inputs = {"messages": [HumanMessage(content=query)]}

    for event in app.stream(inputs, stream_mode="values"):
        last = event["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            print(f"[Tool call requested] {tool_calls}")
        else:
            role = last.type if hasattr(last, "type") else last.__class__.__name__
            print(f"[{role}] {last.content}")
    print("\nDone.")
