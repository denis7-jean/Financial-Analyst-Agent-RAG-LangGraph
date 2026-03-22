from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.state import AgentState, AuditResult, RevisionPlan
from src.nodes.router import ROUTE_LLM


def financial_audit_node(state: AgentState) -> AgentState:
    draft_answer = state.get("draft_answer", "")

    if state.get("route") != "filing_financial" or state.get("revision_count", 0) >= 3:
        return {
            "messages": [AIMessage(content=draft_answer)],
            "audit_feedback": "Approved or Max Retries",
        }

    auditor = ROUTE_LLM.with_structured_output(AuditResult)
    result = auditor.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict financial auditor.\n"
                    "Verify every number in the draft against the retrieved context.\n"
                    "CRITICAL: For tabular evidence, explicitly verify the selected year-to-column mapping before approving the answer.\n"
                    "For tabular evidence, explain your verification in the thought field by mapping headers to row values from left to right.\n"
                    "Only fail the answer when the alignment is actually unsupported or clearly wrong, not merely because the table formatting is imperfect.\n"
                    "If any number is fabricated, hallucinated, or pulled from the wrong year, "
                    "set is_correct to false and provide specific feedback."
                )
            ),
            SystemMessage(content=f"Retrieved context:\n{state.get('retrieval_context', '')}"),
            HumanMessage(content=f"Draft answer:\n{draft_answer}"),
        ]
    )

    if result.is_correct:
        return {
            "messages": [AIMessage(content=draft_answer)],
            "audit_feedback": "Approved",
        }

    return {"audit_feedback": result.feedback}


def narrative_audit_node(state: AgentState) -> AgentState:
    draft_answer = state.get("draft_answer", "")

    if state.get("route") != "filing_narrative" or state.get("revision_count", 0) >= 3:
        return {
            "messages": [AIMessage(content=draft_answer)],
            "audit_feedback": "Approved or Max Retries",
        }

    auditor = ROUTE_LLM.with_structured_output(AuditResult)
    result = auditor.invoke(
        [
            SystemMessage(
                content=(
                    "You are a financial filing auditor reviewing a narrative answer.\n"
                    "Verify that claims in the draft are supported by the retrieved context.\n"
                    "Focus on factual accuracy: are the described risks, strategies, or qualitative statements present in the filing?\n"
                    "Do not fail the answer for style, brevity, or missing detail — only for unsupported or contradicted claims.\n"
                    "If any claim is fabricated or contradicted by the context, "
                    "set is_correct to false and provide specific feedback."
                )
            ),
            SystemMessage(content=f"Retrieved context:\n{state.get('retrieval_context', '')}"),
            HumanMessage(content=f"Draft answer:\n{draft_answer}"),
        ]
    )

    if result.is_correct:
        return {
            "messages": [AIMessage(content=draft_answer)],
            "audit_feedback": "Approved",
        }

    return {"audit_feedback": result.feedback}


def revision_plan_node(state: AgentState) -> AgentState:
    audit_feedback = state.get("audit_feedback", "")
    draft_answer = state.get("draft_answer", "")
    retrieval_context = state.get("retrieval_context", "")

    planner = ROUTE_LLM.with_structured_output(RevisionPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are diagnosing why a financial answer failed audit.\n"
                    "Classify the failure reason and recommend where to retry:\n"
                    "- row_mismatch: wrong row was selected → retry_node: cell_plan_node\n"
                    "- year_mismatch: wrong year column was used → retry_node: cell_plan_node\n"
                    "- citation_mismatch: answer not grounded in evidence → retry_node: draft_answer_node\n"
                    "- insufficient_context: evidence did not contain the answer → retry_node: filing_financial_node\n"
                    "Be conservative: only choose filing_financial_node if context is "
                    "genuinely missing, not just misread."
                )
            ),
            SystemMessage(content=f"Retrieved context:\n{retrieval_context}"),
            SystemMessage(content=f"Draft answer:\n{draft_answer}"),
            HumanMessage(content=f"Audit feedback:\n{audit_feedback}"),
        ]
    )

    return {
        "audit_feedback": plan.suggested_fix,
        "retry_target": plan.retry_node,
    }
