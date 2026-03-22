from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL
from src.core.state import AgentState, DraftAnswer

ANSWER_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def draft_answer_node(state: AgentState) -> AgentState:
    retrieval_context = state.get("retrieval_context", "")
    calculator_result = state.get("calculator_result", "")
    math_expression = state.get("math_expression", "")
    math_operands = state.get("math_operands", [])
    audit_feedback = state.get("audit_feedback", "")

    system_messages = [
        SystemMessage(
            content=(
                "You are a professional financial analyst drafting an answer.\n"
                "You are in the draft answer node of an explicitly routed graph.\n"
                "Do not decide whether arithmetic is needed. That decision has already been made upstream.\n"
                "If a calculator result is provided, you must use that result and must not perform mental math.\n"
                "If filing context is provided, answer from it and cite source, page, and section.\n"
                "Be concise, direct, and analytical."
            )
        )
    ]

    if retrieval_context:
        system_messages.append(
            SystemMessage(content=f"Retrieved context:\n{retrieval_context}")
        )
    if calculator_result:
        system_messages.append(
            SystemMessage(
                content=(
                    f"Calculator expression: {math_expression}\n"
                    f"Calculator operands: {math_operands}\n"
                    f"Calculator result: {calculator_result}\n"
                    "Use this result directly."
                )
            )
        )
    if audit_feedback:
        system_messages.append(
            SystemMessage(
                content=(
                    "Correct your previous draft based on this audit feedback:\n"
                    f"{audit_feedback}"
                )
            )
        )

    system_messages.append(
        SystemMessage(
            content=(
                "CRITICAL: If extracting data from table_evidence, you MUST use the thought field to explicitly map the requested year to the correct column before writing the answer. "
                "Only refuse or hedge if the table alignment is genuinely ambiguous after inspecting the headers and nearby values."
            )
        )
    )

    structured_llm = ANSWER_LLM.with_structured_output(DraftAnswer)
    response = structured_llm.invoke([*system_messages, *state.get("messages", [])])
    new_count = (state.get("revision_count") or 0) + 1
    return {"draft_answer": response.answer, "revision_count": new_count}
