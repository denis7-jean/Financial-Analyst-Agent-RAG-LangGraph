from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.core.state import AgentState, MarketPlan, NewsPlan
from src.nodes.router import COMPANY_TICKER_MAP, ROUTE_LLM, _guess_ticker, _latest_user_message
from src.tools.tools import web_search, yfinance_tool


def market_node(state: AgentState, config: RunnableConfig, store) -> AgentState:
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

    if ticker:
        try:
            user_id = config.get("configurable", {}).get("user_id", "guest")
            namespace = ("tickers", user_id)
            existing = store.get(namespace, ticker)
            if existing:
                count = existing.value.get("count", 0) + 1
                store.put(namespace, ticker, {"ticker": ticker, "count": count})
            else:
                store.put(namespace, ticker, {"ticker": ticker, "count": 1})
        except Exception:
            pass  # never block the main flow

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
