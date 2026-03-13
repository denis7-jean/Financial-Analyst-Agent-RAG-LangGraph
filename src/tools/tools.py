from __future__ import annotations

import json
import math
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from simpleeval import simple_eval

from src.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS
from src.retrieval.retrieval import hybrid_search


SAFE_MATH_PATTERN = re.compile(r"^[0-9\.\+\-\*\/%\(\),\s_a-zA-Z]+$")


def _compact_text(text: str, max_chars: int = 1400) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _citation(document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source", metadata.get("doc_source", "N/A"))
    page = metadata.get("page", "N/A")
    section = metadata.get("section", "N/A")
    return f"{source} | page {page} | section {section}"


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant excerpts found."

    parts = []
    for index, document in enumerate(docs, start=1):
        metadata = document.metadata or {}
        parts.append(
            f"[{index}] {_citation(document)}\n"
            f"chunk_kind: {metadata.get('chunk_kind', 'N/A')}\n"
            f"{_compact_text(document.page_content)}"
        )
    return "\n\n---\n\n".join(parts)


@tool("search_10k", return_direct=False)
def search_10k(query: str) -> str:
    """
    Hybrid retrieval over ingested 10-K chunks.
    Use this tool for filing-backed answers and citations.
    """
    docs = hybrid_search(query=query, k=6, weights=(0.5, 0.5))
    return _format_docs(docs)


@tool("yfinance_tool", return_direct=False)
def yfinance_tool(ticker: str) -> str:
    """
    Fetch a market snapshot for a public ticker using yfinance.
    Returns current price, valuation metrics, and recent price context.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        return "Provide a ticker symbol, for example AAPL or MSFT."

    try:
        import yfinance as yf
    except ImportError:
        return "yfinance is not installed. Add `yfinance` to requirements."

    try:
        security = yf.Ticker(symbol)
        info = getattr(security, "info", {}) or {}
        fast_info = getattr(security, "fast_info", {}) or {}
        history = security.history(period="1mo", interval="1d")

        latest_close = None
        month_change_pct = None
        if history is not None and not history.empty:
            latest_close = float(history["Close"].iloc[-1])
            first_close = float(history["Close"].iloc[0])
            if first_close:
                month_change_pct = ((latest_close - first_close) / first_close) * 100.0

        snapshot = {
            "ticker": symbol,
            "company_name": info.get("shortName") or info.get("longName"),
            "currency": info.get("currency") or fast_info.get("currency"),
            "current_price": info.get("currentPrice") or fast_info.get("lastPrice") or latest_close,
            "previous_close": info.get("previousClose") or fast_info.get("previousClose"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "volume": info.get("volume") or fast_info.get("lastVolume"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or fast_info.get("yearHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow") or fast_info.get("yearLow"),
            "one_month_change_pct": round(month_change_pct, 4) if month_change_pct is not None else None,
        }
        return json.dumps(snapshot, indent=2, default=str)
    except Exception as exc:
        return f"yfinance lookup failed for {symbol}: {exc}"


@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """
    Deterministic calculator for financial arithmetic.
    Accepts raw arithmetic expressions only.
    """
    cleaned = expression.strip()
    if not cleaned:
        return "Provide a math expression, for example (391035-383285)/383285*100."
    if not SAFE_MATH_PATTERN.fullmatch(cleaned):
        return "Unsupported characters in expression."

    normalized = cleaned.replace("^", "**")
    try:
        import numexpr as ne

        result = ne.evaluate(normalized, global_dict={}, local_dict={})
        if hasattr(result, "item"):
            result = result.item()
        return str(result)
    except ImportError:
        math_functions = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        try:
            return str(simple_eval(normalized, functions=math_functions))
        except Exception as exc:
            return f"Error processing calculation: {exc}"
    except Exception as exc:
        return f"Error processing calculation: {exc}"


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """
    Search macroeconomic or market news using Tavily.
    """
    if not query.strip():
        return "Provide a search query."
    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY is not set."

    try:
        from tavily import TavilyClient
    except ImportError:
        return "tavily-python is not installed. Add `tavily-python` to requirements."

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=TAVILY_MAX_RESULTS)
        results = response.get("results", [])
        if not results:
            return "No web results found."

        formatted = []
        for index, item in enumerate(results, start=1):
            formatted.append(
                f"[{index}] {item.get('title', 'Untitled')}\n"
                f"url: {item.get('url', 'N/A')}\n"
                f"{_compact_text(item.get('content', ''), max_chars=600)}"
            )
        return "\n\n---\n\n".join(formatted)
    except Exception as exc:
        return f"Tavily search failed: {exc}"


TOOLS = [search_10k, yfinance_tool, calculator, web_search]


if __name__ == "__main__":
    print("Loaded tools:")
    for tool_obj in TOOLS:
        print(f"- {tool_obj.name}")
