from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


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
