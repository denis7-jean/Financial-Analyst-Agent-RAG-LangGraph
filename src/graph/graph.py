# src/graph/graph.py
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools.tools import TOOLS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# ---------- State Definition ----------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------- LLM with Bound Tools ----------
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)


# ---------- Nodes ----------
def agent(state: AgentState) -> AgentState:
    """Core agent node: runs the LLM over the conversation state."""
    system = SystemMessage(
        content=(
            "You are a professional financial analyst.\n"
            "\n"
            "STRICT RULES:\n"
            "1. If a question involves ANY arithmetic (growth rates, differences, projections), "
            "you MUST call the calculator tool.\n"
            "2. Do NOT do mental math or inline calculations.\n"
            "3. Always present numbers with commas and a space before units "
            "(e.g., '391,035 million USD').\n"
            "4. When calculating, output in the following format:\n"
            "   - Formula\n"
            "   - Substitution\n"
            "   - Final numeric result\n"
            "5. Never merge numbers and units.\n"
            "6. When calling calculator, pass ONLY a raw expression like 391035*1.15 or 449690.25-383285 (no commas, no words).\n"
            "\n"
            "If you violate these rules, the answer is considered incorrect."
        )
    )

    messages = state["messages"]

    # Avoid repeatedly injecting the system message every turn
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(TOOLS)


# ---------- Routing Logic ----------
def should_continue(state: AgentState):
    """Decide whether to call tools or end the conversation."""
    last_message = state["messages"][-1]
    # If the LLM requested tool calls, go to the tools node; otherwise, end.
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------- Build the Graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compiled application
app = workflow.compile()


if __name__ == "__main__":
    # Complex query requiring retrieval + math
    query = (
        "What was Apple's total net sales in 2024? "
        "If I assume a 5% increase for next year, what would that be?"
    )

    print("Streaming agent execution:\n")
    inputs = {"messages": [HumanMessage(content=query)]}

    # Stream the steps so we can see tool calls and responses
    for event in app.stream(inputs, stream_mode="values"):
        # Each event is a partial state; we show only the latest message content/tool call.
        last = event["messages"][-1]
        # Pretty-print based on message type
        if getattr(last, "tool_calls", None):
            print(f"[Tool call requested] {last.tool_calls}")
        else:
            role = last.type if hasattr(last, 'type') else last.__class__.__name__
            print(f"[{role}] {last.content}")
    print("\nDone.")
