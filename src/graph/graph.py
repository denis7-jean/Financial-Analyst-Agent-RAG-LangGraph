from __future__ import annotations

import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.config import GEMINI_MODEL
from src.tools.tools import calculator, search_10k, web_search, yfinance_tool


class RouteQuery(BaseModel):
    route: Literal["filing", "market", "news", "general"] = Field(
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


class MathPlan(BaseModel):
    operands: list[str] = Field(
        default_factory=list,
        description="Exact numeric operands extracted from the retrieval context for the calculation.",
    )
    expression: str = Field(
        ...,
        description="Strict raw arithmetic expression using only numbers, parentheses, and operators.",
    )


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    retrieval_context: str
    retrieval_source: str
    math_operands: list[str]
    math_expression: str
    calculator_result: str


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
    uppercase_candidates = re.findall(r"\b[A-Z]{1,5}\b", text)
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
                    "- filing: SEC filing facts, risk sections, year-over-year filing comparisons, or anything that should be answered from the 10-K.\n"
                    "- market: stock price, valuation, ticker, or public market data.\n"
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
    }


def filing_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    planner = ROUTE_LLM.with_structured_output(FilingPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Generate a retrieval query for the 10-K search tool.\n"
                    "If the user asks for a comparison across multiple years, do not simply rewrite the question.\n"
                    "You must generate a dense keyword-heavy retrieval query formatted like:\n"
                    "[Company] [Target Metric] [Year 1] [Year 2] consolidated statements of operations\n"
                    "Example: Apple total net sales 2023 2024 consolidated statements of operations\n"
                    "Preserve the company, target metric, and every year explicitly mentioned in the user query."
                )
            ),
            HumanMessage(content=latest_question),
        ]
    )
    result = search_10k.invoke(plan.search_query)
    return {"retrieval_context": result, "retrieval_source": "filing"}


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


def requires_math(state: AgentState) -> str:
    question = _latest_user_message(state).lower()
    years = _extract_years(question)

    if any(keyword in question for keyword in MATH_KEYWORDS):
        return "math_prep_node"

    if len(set(years)) >= 2 and any(
        phrase in question
        for phrase in ["how much", "differs", "difference", "higher", "lower", "more", "less"]
    ):
        return "math_prep_node"

    return "generate_answer_node"


def math_prep_node(state: AgentState) -> AgentState:
    latest_question = _latest_user_message(state)
    retrieval_context = state.get("retrieval_context", "")

    extractor = ROUTE_LLM.with_structured_output(MathPlan)
    math_plan = extractor.invoke(
        [
            SystemMessage(
                content=(
                    "You are preparing a calculator call.\n"
                    "Read the retrieved filing context and the user's question.\n"
                    "Extract the exact numeric operands needed for the requested comparison or change.\n"
                    "Construct a strict mathematical expression using only numbers, decimals, parentheses, and arithmetic operators.\n"
                    "Do not answer the question. Do not summarize. Return only the operands and the expression."
                )
            ),
            SystemMessage(content=f"Retrieved filing context:\n{retrieval_context}"),
            HumanMessage(content=latest_question),
        ]
    )

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


def generate_answer_node(state: AgentState) -> AgentState:
    retrieval_context = state.get("retrieval_context", "")
    calculator_result = state.get("calculator_result", "")
    math_expression = state.get("math_expression", "")
    math_operands = state.get("math_operands", [])

    system_messages = [
        SystemMessage(
            content=(
                "You are a professional financial analyst.\n"
                "You are in the final answer node of an explicitly routed graph.\n"
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

    response = ANSWER_LLM.invoke([*system_messages, *state.get("messages", [])])
    return {"messages": [response]}


def route_after_router(state: AgentState) -> str:
    return state.get("route", "general")


workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("filing_node", filing_node)
workflow.add_node("market_node", market_node)
workflow.add_node("news_node", news_node)
workflow.add_node("math_prep_node", math_prep_node)
workflow.add_node("calculator_node", calculator_node)
workflow.add_node("generate_answer_node", generate_answer_node)

workflow.add_edge(START, "router_node")
workflow.add_conditional_edges(
    "router_node",
    route_after_router,
    {
        "filing": "filing_node",
        "market": "market_node",
        "news": "news_node",
        "general": "generate_answer_node",
    },
)
workflow.add_conditional_edges(
    "filing_node",
    requires_math,
    {
        "math_prep_node": "math_prep_node",
        "generate_answer_node": "generate_answer_node",
    },
)
workflow.add_edge("math_prep_node", "calculator_node")
workflow.add_edge("calculator_node", "generate_answer_node")
workflow.add_edge("market_node", "generate_answer_node")
workflow.add_edge("news_node", "generate_answer_node")
workflow.add_edge("generate_answer_node", END)

app = workflow.compile()


if __name__ == "__main__":
    query = "Compare Apple's 2023 and 2024 net sales and tell me how much higher 2024 was."
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "graph-math-smoke-test"}},
    )
    final_message = result["messages"][-1]
    print(getattr(final_message, "content", final_message))
