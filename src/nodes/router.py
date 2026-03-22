from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL
from src.core.state import AgentState, RouteQuery

ROUTE_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

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


def router_node(state: AgentState, config: RunnableConfig, store) -> AgentState:
    latest_question = _latest_user_message(state)

    remembered_tickers = []
    try:
        user_id = config.get("configurable", {}).get("user_id", "guest")
        namespace = ("tickers", user_id)
        memories = store.search(namespace)
        remembered_tickers = [m.value["ticker"] for m in memories]
    except Exception:
        pass

    memory_hint = ""
    if remembered_tickers:
        memory_hint = (
            f"\nUser has previously asked about: "
            f"{', '.join(remembered_tickers)}. "
            f"If the query is ambiguous, prefer these tickers."
        )

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
                    + memory_hint
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
