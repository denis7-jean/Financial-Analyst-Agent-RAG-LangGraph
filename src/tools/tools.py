# src/tools/tools.py
from __future__ import annotations

import math
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

from src.retrieval.retrieval import get_retriever


def get_financial_retriever_tool():
    """
    Returns a retriever tool that searches the 2024 Apple 10-K filings.
    """
    retriever = get_retriever()
    return create_retriever_tool(
        retriever=retriever,
        name="search_10k",
        description="Searches and returns excerpts from the 2024 Apple 10-K financial report.",
    )


@tool(
    "calculator",
    return_direct=True,
)
def calculator(expression: str) -> str:
    """Useful for performing mathematical calculations. Input should be a valid mathematical expression string."""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOLS = [get_financial_retriever_tool(), calculator]


if __name__ == "__main__":
    print("Loaded tools:")
    for t in TOOLS:
        print(f"- {t.name}")

    test_expr = "50 * 25"
    print(f"\nTesting calculator with '{test_expr}':")
    print(calculator.invoke(test_expr))
