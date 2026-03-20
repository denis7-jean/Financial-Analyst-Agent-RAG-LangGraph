from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.config import GEMINI_MODEL
from src.tools.tools import calculator, search_10k, web_search, yfinance_tool


class RouteQuery(BaseModel):
    route: Literal["filing_financial", "filing_narrative", "market", "news", "general"] = Field(
        ...,
        description="Best route for answering the user's latest question.",
    )


class FilingPlan(BaseModel):
    search_query: str = Field(..., description="Focused retrieval query for the 10-K search tool.")


class NewsPlan(BaseModel):
    search_query: str = Field(..., description="Focused macro or news query for web search.")


class MarketPlan(BaseModel):
    ticker: str | None = Field(default=None, description="Public ticker symbol when identifiable.")
    company_name: str | None = Field(default=None, description="Company name if a ticker is not explicit.")


class TableSelection(BaseModel):
    row_label: str = Field(..., description="The exact label of the row being extracted (e.g., 'Total net sales').")
    requested_year: str = Field(..., description="The year requested by the user.")
    selected_column_index: int = Field(..., description="The index of the column corresponding to the requested year (e.g., 0 for the first data column).")
    selected_value: str = Field(..., description="The exact numerical value extracted from that column.")


class SelectedCell(BaseModel):
    row_label: str = Field(..., description="Exact row label from the table (e.g., 'Total net sales').")
    column_header: str = Field(..., description="Column header, typically a year (e.g., '2024').")
    value: str = Field(..., description="The numeric value at this cell.")


class CellPlan(BaseModel):
    thought: str = Field(..., description="Brief reasoning about which cells to select and why.")
    cells: list[SelectedCell] = Field(
        default_factory=list,
        description="Ordered list of cells selected from the table evidence.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="'high' only when every cell maps unambiguously to a row+column in the evidence.",
    )


class MathPlan(BaseModel):
    thought: str = Field(..., description="Brief verification logic mapping headers to values.")
    selections: list[TableSelection] = Field(
        default_factory=list,
        description="Explicit mapping of the evidence used for the calculation.",
    )
    operands: list[str] = Field(
        ...,
        description="Exact numeric operands extracted.",
    )
    expression: str = Field(
        ...,
        description="Strict raw arithmetic expression.",
    )
    evidence_complete: bool = Field(
        ...,
        description="Set to False if year-to-column alignment is ambiguous or missing.",
    )
    target_metric: str = Field(
        ...,
        description="The specific financial metric being looked up (e.g., 'Total net sales').",
    )
    row_match_confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence that the correct row was identified. 'high' only when the row label is an exact or near-exact match.",
    )


class DraftAnswer(BaseModel):
    thought: str = Field(
        ...,
        description="Step-by-step reasoning. If using table evidence, explicitly map the requested year to the most likely correct column before writing the answer.",
    )
    answer: str = Field(..., description="The final concise answer to the user.")


class AuditResult(BaseModel):
    thought: str = Field(
        ...,
        description="Step-by-step verification logic. For tabular evidence, explicitly map the year headers to the row values from left to right before deciding.",
    )
    is_correct: bool = Field(
        ...,
        description="True if all numbers are factually supported by the correct year column.",
    )
    feedback: str = Field(
        ...,
        description="If false, provide specific instructions on what is wrong.",
    )


class RevisionPlan(BaseModel):
    retry_reason: Literal[
        "row_mismatch",
        "year_mismatch",
        "citation_mismatch",
        "insufficient_context",
    ] = Field(..., description="Classification of why the audit failed.")
    suggested_fix: str = Field(..., description="Specific guidance for the retry node.")
    retry_node: Literal[
        "cell_plan_node",
        "draft_answer_node",
        "filing_financial_node",
    ] = Field(..., description="Which node should handle the retry.")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    retrieval_context: str
    retrieval_source: str
    math_operands: list[str]
    math_expression: str
    calculator_result: str
    draft_answer: str
    audit_feedback: str
    revision_count: int
    row_match_confidence: str
    cell_plan: dict[str, Any]
    retry_target: str


ROUTE_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
MATH_PREP_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0).bind_tools([calculator])
ANSWER_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

COMPANY_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "nvidia": "NVDA",
}
MATH_KEYWORDS = [
    "compare",
    "difference",
    "higher",
    "increase",
    "decrease",
    "growth",
    "versus",
    "vs",
    "change",
    "delta",
    "yoy",
    "year-over-year",
    "how much more",
    "how much less",
]


def _latest_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _guess_ticker(text: str) -> str | None:
    TICKER_BLOCKLIST = {
        "I", "A", "IN", "IF", "THE", "FOR", "AND", "OR",
        "OF", "TO", "IS", "IT", "AT", "AN", "AS", "BE",
        "BY", "DO", "GO", "HE", "ME", "MY", "NO", "ON",
        "SO", "UP", "US", "WE",
    }
    uppercase_candidates = [
        t for t in re.findall(r"\b[A-Z]{1,5}\b", text)
        if t not in TICKER_BLOCKLIST
    ]
    if uppercase_candidates:
        return uppercase_candidates[0]

    lowered = text.lower()
    for company_name, ticker in COMPANY_TICKER_MAP.items():
        if company_name in lowered:
            return ticker
    return None


def _extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def router_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    router = ROUTE_LLM.with_structured_output(RouteQuery)
    route = router.invoke(
        [
            SystemMessage(
                content=(
                    "Classify the user's latest question into exactly one route.\n"
                    "- filing_financial: historical financial metrics from SEC filings — net sales, revenue, gross margin, net income, cash flow, EPS from annual reports. Examples: 'What were Apple's net sales in 2024?', 'What was Apple's gross margin last year?'\n"
                    "- filing_narrative: qualitative sections like risk factors, business descriptions, MD&A discussion, or prose from the 10-K.\n"
                    "- market: stock price, current valuation, P/E ratio, market cap, ticker data, or any question about live/current market data. Examples: 'What is Apple's stock price?', 'What is AAPL trading at?', 'What is Apple's market cap today?'\n"
                    "- news: macro, economic, geopolitical, or current-events search.\n"
                    "- general: conversational or generic questions that do not require an external tool."
                )
            ),
            HumanMessage(content=latest_question),
        ]
    )
    return {
        "route": route.route,
        "retrieval_context": "",
        "retrieval_source": route.route,
        "math_operands": [],
        "math_expression": "",
        "calculator_result": "",
        "draft_answer": "",
        "audit_feedback": "",
        "revision_count": 0,
    }


def filing_financial_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    planner = ROUTE_LLM.with_structured_output(FilingPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Generate a retrieval query for the 10-K search tool.\n"
                    "The user is asking about specific financial metrics or numbers.\n"
                    "You must generate a dense keyword-heavy retrieval query formatted like:\n"
                    "[Company] [Target Metric] [Year 1] [Year 2] consolidated statements of operations\n"
                    "Example: Apple total net sales 2023 2024 consolidated statements of operations\n"
                    "Preserve the company, target metric, and every year explicitly mentioned in the user query."
                )
            ),
            HumanMessage(content=latest_question),
        ]
    )
    result = search_10k.invoke({"query": plan.search_query, "mode": "financial"})
    return {"retrieval_context": result, "retrieval_source": "filing_financial"}


def cell_plan_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    retrieval_context = state.get("retrieval_context", "")

    # Extract row_candidates from the structured evidence JSON
    row_candidates_text = "[]"
    try:
        evidence = json.loads(retrieval_context)
        row_candidates = evidence.get("row_candidates", [])
        if row_candidates:
            row_candidates_text = json.dumps(row_candidates, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass

    planner = ROUTE_LLM.with_structured_output(CellPlan)
    cell_plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are selecting specific table cells to answer a financial question.\n"
                    "Read the retrieved filing context and the pre-selected row candidates below.\n"
                    "For each value the question requires, identify the exact row label, column header (year), and numeric value.\n"
                    "Set confidence to 'high' only if every cell maps unambiguously to a row+column in the evidence.\n"
                    "If the row candidates are empty or ambiguous, inspect the candidate_tables directly.\n"
                    "Do not answer the question. Return only the cell selections."
                )
            ),
            SystemMessage(content=f"Retrieved filing context:\n{retrieval_context}"),
            SystemMessage(content=f"Pre-selected row candidates:\n{row_candidates_text}"),
            HumanMessage(content=latest_question),
        ]
    )

    return {
        "cell_plan": {
            "thought": cell_plan.thought,
            "cells": [c.model_dump() for c in cell_plan.cells],
            "confidence": cell_plan.confidence,
        },
        "row_match_confidence": cell_plan.confidence,
    }


def filing_narrative_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    planner = ROUTE_LLM.with_structured_output(FilingPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Generate a retrieval query for the 10-K search tool.\n"
                    "The user is asking about qualitative information, risk factors, "
                    "business descriptions, or narrative sections.\n"
                    "Generate a keyword query that targets the relevant section of the 10-K."
                )
            ),
            HumanMessage(content=latest_question),
        ]
    )
    result = search_10k.invoke({"query": plan.search_query, "mode": "narrative"})
    return {"retrieval_context": result, "retrieval_source": "filing_narrative"}


def market_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    planner = ROUTE_LLM.with_structured_output(MarketPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Extract the most likely public ticker symbol from the user's question. "
                    "If a ticker is not explicit, infer it from the company name when obvious."
                )
            ),
            HumanMessage(content=latest_question),
        ]
    )

    ticker = plan.ticker or _guess_ticker(latest_question)
    if not ticker and plan.company_name:
        ticker = COMPANY_TICKER_MAP.get(plan.company_name.lower())

    if not ticker:
        context = "Unable to determine a ticker symbol from the user's request."
    else:
        context = yfinance_tool.invoke(ticker)

    return {"retrieval_context": context, "retrieval_source": "market"}


def news_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    planner = ROUTE_LLM.with_structured_output(NewsPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content="Rewrite the user's question into a concise web search query for macroeconomic or news research."
            ),
            HumanMessage(content=latest_question),
        ]
    )
    result = web_search.invoke(plan.search_query)
    return {"retrieval_context": result, "retrieval_source": "news"}


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


def route_after_router(state: AgentState) -> str:
    return state.get("route", "general")


def check_math_confidence(state: AgentState) -> str:
    if state.get("row_match_confidence") == "high":
        return "calculator_node"
    return "draft_answer_node"


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


def route_revision(state: AgentState) -> str:
    target = state.get("retry_target", "")
    if target in ("cell_plan_node", "draft_answer_node", "filing_financial_node"):
        return target
    return "draft_answer_node"


workflow = StateGraph(AgentState)
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

checkpointer = MemorySaver()
store = InMemoryStore()
app = workflow.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    query = "Compare Apple's 2023 and 2024 net sales and tell me how much higher 2024 was."
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "graph-math-smoke-test"}},
    )
    final_message = result["messages"][-1]
    print(getattr(final_message, "content", final_message))
