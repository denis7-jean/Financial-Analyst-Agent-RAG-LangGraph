from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.state import AgentState, CellPlan, FilingPlan
from src.nodes.router import ROUTE_LLM, _latest_user_message
from src.tools.tools import search_10k


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
