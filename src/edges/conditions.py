from __future__ import annotations

from src.core.state import AgentState
from src.nodes.router import MATH_KEYWORDS, _extract_years, _latest_user_message


def _needs_math(state: AgentState) -> bool:
    """Helper: detect whether the question requires arithmetic."""
    question = _latest_user_message(state).lower()
    years = _extract_years(question)

    if any(keyword in question for keyword in MATH_KEYWORDS):
        return True

    if len(set(years)) >= 2 and any(
        phrase in question
        for phrase in ["how much", "differs", "difference", "higher", "lower", "more", "less"]
    ):
        return True

    return False


def requires_math(state: AgentState) -> str:
    """Conditional edge: filing_financial_node → cell_plan_node or draft_answer_node."""
    if _needs_math(state):
        return "cell_plan_node"
    return "draft_answer_node"


def check_cell_confidence(state: AgentState) -> str:
    """Conditional edge: cell_plan_node → math_prep_node (high) or draft_answer_node."""
    confidence = (state.get("cell_plan") or {}).get("confidence", "low")
    if confidence == "high":
        return "math_prep_node"
    return "draft_answer_node"


def check_math_confidence(state: AgentState) -> str:
    if state.get("row_match_confidence") == "high":
        return "calculator_node"
    return "draft_answer_node"


def route_after_router(state: AgentState) -> str:
    return state.get("route", "general")


def route_to_audit(state: AgentState) -> str:
    route = state.get("route", "")
    if route == "filing_financial":
        return "financial_audit_node"
    if route == "filing_narrative":
        return "narrative_audit_node"
    # Non-filing routes pass through narrative_audit_node (auto-approves)
    return "narrative_audit_node"


def check_audit(state: AgentState) -> str:
    feedback = state.get("audit_feedback", "")
    if feedback.startswith("Approved") or feedback.startswith("Max"):
        return "END"
    return "revision_plan_node"


def route_revision(state: AgentState) -> str:
    target = state.get("retry_target", "")
    if target in ("cell_plan_node", "draft_answer_node", "filing_financial_node"):
        return target
    return "draft_answer_node"
