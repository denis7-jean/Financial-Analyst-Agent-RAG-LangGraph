from __future__ import annotations

import json
import os
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in constrained environments
    def load_dotenv(*args, **kwargs):
        return False

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
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
    {
        "question": "What was Apple's gross margin in 2024?",
        "expected_answer": "180,683",
    },
    {
        "question": "What was Apple's net income in 2024?",
        "expected_answer": "93,736",
    },
    {
        "question": "What does Apple identify as its primary competition risks?",
        "expected_answer": "Risk Factors",
    },
    {
        "question": "What was Apple's cash flow from operations in 2024?",
        "expected_answer": "118,254",
    },
]


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


_FILING_KEYWORDS = ("net sales", "revenue", "income", "margin", "cash flow", "eps", "profit", "operating")
_NARRATIVE_KEYWORDS = ("risk factors", "risk", "business description", "md&a", "management discussion")


def evaluator_route_correct(run, example) -> dict[str, Any]:
    """Check if the agent used the right route type for the question."""
    predicted_answer = (run.outputs or {}).get("answer", "").lower()
    expected_answer = (example.outputs or {}).get("expected_answer", "").lower()

    expects_narrative = any(kw in expected_answer for kw in _NARRATIVE_KEYWORDS)
    expects_financial = any(kw in expected_answer for kw in _FILING_KEYWORDS) or any(
        ch.isdigit() for ch in expected_answer
    )

    if expects_narrative:
        score = 1.0 if any(kw in predicted_answer for kw in _NARRATIVE_KEYWORDS) else 0.0
    elif expects_financial:
        score = 1.0 if any(ch.isdigit() for ch in predicted_answer) else 0.0
    else:
        score = 1.0  # general — pass through

    return {"key": "route_correct", "score": score}


def evaluator_retrieval_support(run, example) -> dict[str, Any]:
    """Check if the answer contains a citation (page, source, section reference)."""
    predicted_answer = (run.outputs or {}).get("answer", "").lower()
    has_citation = any(
        marker in predicted_answer
        for marker in ("page", "aapl", "apple", "risk factors", "10-k", "10k", "filing")
    )
    return {"key": "retrieval_support_present", "score": 1.0 if has_citation else 0.0}


def evaluator_cell_selection(run, example) -> dict[str, Any]:
    """For numeric questions, check if the correct numbers appear in the answer."""
    predicted_answer = (run.outputs or {}).get("answer", "")
    expected_answer = (example.outputs or {}).get("expected_answer", "")

    # Non-numeric questions: pass through
    if not any(ch.isdigit() for ch in expected_answer):
        return {"key": "cell_selection_correct", "score": 1.0, "comment": "Non-numeric question, skipped."}

    judge_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    try:
        response = judge_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Does the predicted answer contain the correct numbers from the expected answer? "
                        "Score 1 if all key numbers match, 0 otherwise. "
                        'Output JSON with keys "score" (int 1 or 0) and "reason" (short explanation).'
                    )
                ),
                HumanMessage(
                    content=f"EXPECTED_ANSWER:\n{expected_answer}\n\nPREDICTED_ANSWER:\n{predicted_answer}"
                ),
            ]
        )
        parsed = json.loads(response.content if isinstance(response.content, str) else json.dumps(response.content))
        return {
            "key": "cell_selection_correct",
            "score": float(int(parsed.get("score", 0))),
            "comment": str(parsed.get("reason", "")),
        }
    except Exception as exc:
        return {"key": "cell_selection_correct", "score": 0.0, "comment": f"Judge failed: {exc}"}


def evaluator_final_answer(run, example) -> dict[str, Any]:
    """GPT-4o judge for overall answer correctness (renamed from llm_correctness_evaluator)."""
    judge_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    predicted_answer = (run.outputs or {}).get("answer", "")
    expected_answer = (example.outputs or {}).get("expected_answer", "")

    system_message = SystemMessage(
        content=(
            "You are an impartial and strict financial grader. Compare the PREDICTED_ANSWER against the EXPECTED_ANSWER.\n"
            "Rubric:\n"
            "a. If the EXPECTED_ANSWER contains specific numbers, the PREDICTED_ANSWER must contain those exact numbers or their mathematical equivalents.\n"
            "b. Ignore conversational filler or extra context in the PREDICTED_ANSWER, as long as the core fact/number matches the EXPECTED_ANSWER.\n"
            "c. You must output a JSON object with strictly two keys: \"score\" (integer 1 if correct, 0 if incorrect) and \"reason\" (short explanation)."
        )
    )
    human_message = HumanMessage(
        content=(
            f"EXPECTED_ANSWER:\n{expected_answer}\n\n"
            f"PREDICTED_ANSWER:\n{predicted_answer}"
        )
    )

    try:
        response = judge_llm.invoke([system_message, human_message])
        raw_content = response.content
        if isinstance(raw_content, str):
            parsed = json.loads(raw_content)
        else:
            parsed = json.loads(json.dumps(raw_content))

        parsed_score = int(parsed.get("score", 0))
        parsed_reason = str(parsed.get("reason", "No reason provided."))
        return {
            "key": "final_answer_correct",
            "score": float(parsed_score),
            "comment": parsed_reason,
        }
    except Exception as exc:
        return {
            "key": "final_answer_correct",
            "score": 0.0,
            "comment": f"Judge evaluation failed: {exc}",
        }


def run_evaluation(client: Client, dataset_name: str = DATASET_NAME):
    return evaluate(
        run_agent,
        data=dataset_name,
        evaluators=[
            evaluator_route_correct,
            evaluator_retrieval_support,
            evaluator_cell_selection,
            evaluator_final_answer,
        ],
        experiment_prefix="financial-analyst-agent",
        description="4-dimension LangSmith evaluation for the Apple 10-K LangGraph agent.",
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
