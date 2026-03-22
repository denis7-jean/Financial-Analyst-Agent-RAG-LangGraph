from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL
from src.core.state import AgentState, MathPlan
from src.nodes.router import ROUTE_LLM, _latest_user_message
from src.tools.tools import calculator

MATH_PREP_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0).bind_tools([calculator])


def math_prep_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    retrieval_context = state.get("retrieval_context", "")
    cell_plan = state.get("cell_plan") or {}

    # Build cell context hint from upstream cell_plan_node
    cell_hint = ""
    if cell_plan.get("cells"):
        cells_summary = "; ".join(
            f"{c['row_label']} ({c['column_header']}): {c['value']}"
            for c in cell_plan["cells"]
        )
        cell_hint = (
            f"\nUpstream cell selections (use these as your operands): {cells_summary}\n"
            f"Cell plan confidence: {cell_plan.get('confidence', 'unknown')}\n"
        )

    extractor = ROUTE_LLM.with_structured_output(MathPlan)
    math_plan = extractor.invoke(
        [
            SystemMessage(
                content=(
                    "You are preparing a calculator call.\n"
                    "Read the retrieved filing context and the user's question.\n"
                    "Extract the exact numeric operands needed for the requested comparison or change.\n"
                    "Construct a strict mathematical expression using only numbers, decimals, parentheses, and arithmetic operators.\n"
                    "CRITICAL: If cell selections are provided below, use those exact values as your operands. "
                    "If extracting from tables without cell selections, you must explicitly map the requested year to the correct column index in the selections list before writing the expression. If the table alignment is unclear, set evidence_complete to False.\n"
                    "Identify the target_metric (the exact row label you are extracting from the table).\n"
                    "Rate row_match_confidence as 'high' only if the row label is an exact or near-exact match to a row in the table. Use 'medium' or 'low' otherwise.\n"
                    "Do not answer the question. Do not summarize. Return only the operands and the expression."
                    + cell_hint
                )
            ),
            SystemMessage(content=f"Retrieved filing context:\n{retrieval_context}"),
            HumanMessage(content=latest_question),
        ]
    )

    # Hard gate: block calculator if row match confidence is not high
    if math_plan.row_match_confidence != "high":
        return {
            "math_operands": math_plan.operands,
            "math_expression": math_plan.expression,
            "row_match_confidence": math_plan.row_match_confidence,
            "calculator_result": "",
        }

    calculator_call = MATH_PREP_LLM.invoke(
        [
            SystemMessage(
                content=(
                    "You must call the calculator tool using the exact expression provided by the user. "
                    "Do not answer in natural language."
                )
            ),
            HumanMessage(content=f"Use this exact expression: {math_plan.expression}"),
        ]
    )

    if not getattr(calculator_call, "tool_calls", None):
        calculator_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": math_plan.expression},
                    "id": "forced_calculator_call",
                    "type": "tool_call",
                }
            ],
        )

    return {
        "messages": [calculator_call],
        "math_operands": math_plan.operands,
        "math_expression": math_plan.expression,
        "row_match_confidence": math_plan.row_match_confidence,
        "calculator_result": "",
    }


def calculator_node(state: AgentState) -> AgentState:
    expression = state.get("math_expression", "")
    result = calculator.invoke(expression)

    tool_call_id = "forced_calculator_call"
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        tool_call_id = tool_calls[0].get("id", tool_call_id)

    tool_message = ToolMessage(
        content=str(result),
        name="calculator",
        tool_call_id=tool_call_id,
    )
    return {
        "messages": [tool_message],
        "calculator_result": str(result),
    }
