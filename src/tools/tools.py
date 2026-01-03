# src/tools/tools.py
from simpleeval import simple_eval
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


@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Useful for performing mathematical calculations."""
    # 1. Define math functions (to supportsqrt, sin, pow, etc)
    math_functions = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    
    try:
        # 2.  math to simple_eval, safe calculation
        result = simple_eval(expression, functions=math_functions)
        return str(result)
    except Exception as e:
        return f"Error processing calculation: {e}"

TOOLS = [get_financial_retriever_tool(), calculator]


if __name__ == "__main__":
    print("Loaded tools:")
    for t in TOOLS:
        print(f"- {t.name}")

    test_expr = "50 * 25"
    print(f"\nTesting calculator with '{test_expr}':")
    print(calculator.invoke(test_expr))
