from __future__ import annotations

import os
import re
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in constrained environments
    def load_dotenv(*args, **kwargs):
        return False

from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from src.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
)
from src.graph.graph import app


load_dotenv()


DATASET_NAME = "financial-analyst-agent-apple-10k"
DATASET_DESCRIPTION = "Small LangSmith dataset for filing-grounded and calculation-focused Apple 10-K evaluation."
DATASET_EXAMPLES = [
    {
        "question": "What were Apple's total net sales in 2024?",
        "expected_answer": "391,035",
    },
    {
        "question": "If Apple's 2024 net sales of 391,035 increased by 5%, what would the projected sales be?",
        "expected_answer": "410,586.75",
    },
    {
        "question": "What were Apple's total net sales in 2023, and how much higher were 2024 net sales than 2023?",
        "expected_answer": "2023 net sales were 383,285 and the difference versus 2024 was 7,750.",
    },
    {
        "question": "What section should I inspect for Apple's major business and operational risks?",
        "expected_answer": "Risk Factors",
    },
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def get_client() -> Client:
    if not LANGCHAIN_API_KEY:
        raise ValueError("LANGCHAIN_API_KEY is not set.")

    os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

    return Client(api_key=LANGCHAIN_API_KEY, api_url=LANGCHAIN_ENDPOINT)


def get_or_create_dataset(client: Client, dataset_name: str = DATASET_NAME):
    existing = next(client.list_datasets(dataset_name=dataset_name), None)
    if existing:
        return existing
    return client.create_dataset(
        dataset_name=dataset_name,
        description=DATASET_DESCRIPTION,
    )


def sync_dataset_examples(client: Client, dataset) -> None:
    desired_examples = {
        item["question"]: item["expected_answer"]
        for item in DATASET_EXAMPLES
    }
    existing_examples = {
        example.inputs.get("question"): example
        for example in client.list_examples(dataset_id=dataset.id)
        if example.inputs.get("question")
    }

    for question, expected_answer in desired_examples.items():
        existing = existing_examples.pop(question, None)
        if existing and (existing.outputs or {}).get("expected_answer") == expected_answer:
            continue
        if existing:
            client.delete_example(existing.id)
        client.create_example(
            dataset_id=dataset.id,
            inputs={"question": question},
            outputs={"expected_answer": expected_answer},
        )

    for stale_example in existing_examples.values():
        client.delete_example(stale_example.id)


def run_agent(inputs: dict[str, Any]) -> dict[str, str]:
    question = inputs["question"]
    thread_id = f"langsmith-eval-{uuid.uuid4().hex}"
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    final_message = result["messages"][-1]
    answer = getattr(final_message, "content", str(final_message))
    return {"answer": answer}


def exact_match_evaluator(run, example) -> dict[str, Any]:
    predicted = _normalize_text((run.outputs or {}).get("answer", ""))
    expected = _normalize_text((example.outputs or {}).get("expected_answer", ""))
    return {
        "key": "exact_match",
        "score": 1.0 if predicted == expected else 0.0,
    }


def contains_expected_answer_evaluator(run, example) -> dict[str, Any]:
    predicted = _normalize_text((run.outputs or {}).get("answer", ""))
    expected = _normalize_text((example.outputs or {}).get("expected_answer", ""))
    return {
        "key": "contains_expected_answer",
        "score": 1.0 if expected and expected in predicted else 0.0,
    }


def run_evaluation(client: Client, dataset_name: str = DATASET_NAME):
    return evaluate(
        run_agent,
        data=dataset_name,
        evaluators=[exact_match_evaluator, contains_expected_answer_evaluator],
        experiment_prefix="financial-analyst-agent",
        description="Baseline LangSmith evaluation for the Apple 10-K LangGraph agent.",
    )


def main() -> int:
    client = get_client()
    dataset = get_or_create_dataset(client)
    sync_dataset_examples(client, dataset)

    results = run_evaluation(client, dataset_name=dataset.name)
    print(f"LangSmith evaluation launched for dataset: {dataset.name}")
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
