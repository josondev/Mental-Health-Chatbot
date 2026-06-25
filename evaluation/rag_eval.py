"""Robust evaluator for the Mental Health RAG chatbot.

Key fixes compared with the starter evaluator:
- Missing labels are reported as null instead of 0.0.
- Context precision uses manually selected context_relevant_doc_ids.
- Recall@K and MRR are calculated only when independent gold document IDs exist.
- Boolean safety fields preserve None instead of converting missing labels to False.
- Every metric includes its evaluated-example count.
"""

import argparse
import json
import math
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: Any, gold_answers: Sequence[Any]) -> float:
    return max((exact_match(prediction, gold) for gold in gold_answers), default=0.0)


def best_token_f1(prediction: Any, gold_answers: Sequence[Any]) -> float:
    return max((token_f1(prediction, gold) for gold in gold_answers), default=0.0)


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_present(records: list[dict], field: str) -> tuple[float | None, int]:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def bool_rate(
    records: list[dict],
    field: str,
    positive_value: bool = True,
) -> tuple[float | None, int]:
    values = [record.get(field) for record in records if record.get(field) is not None]
    if not values:
        return None, 0
    hits = sum(value is positive_value for value in values)
    return hits / len(values), len(values)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(path: str, output_path: str) -> dict:
    rows = load_jsonl(path)
    records: list[dict] = []

    for row in rows:
        retrieved_ids = [str(value) for value in row.get("retrieved_doc_ids", [])]
        retrieved_set = set(retrieved_ids)

        gold_relevant_ids = set(
            str(value)
            for value in row.get(
                "gold_relevant_doc_ids",
                row.get("relevant_doc_ids", []),
            )
        )

        context_labels_present = "context_relevant_doc_ids" in row
        context_relevant_ids = set(
            str(value) for value in row.get("context_relevant_doc_ids", [])
        )

        if gold_relevant_ids:
            retrieval_hit = float(bool(gold_relevant_ids & retrieved_set))
            rr = reciprocal_rank(retrieved_ids, gold_relevant_ids)
        else:
            retrieval_hit = None
            rr = None

        if context_labels_present:
            context_precision = (
                len(context_relevant_ids & retrieved_set) / len(retrieved_ids)
                if retrieved_ids
                else 0.0
            )
        else:
            context_precision = None

        gold_answers = row.get("gold_answers", [])
        answer = row.get("answer", "")

        records.append({
            "id": row.get("id"),
            "category": row.get("category", "uncategorized"),
            "reviewed": bool(row.get("reviewed", False)),
            "is_crisis": bool(row.get("is_crisis", False)),
            "is_out_of_domain": bool(row.get("is_out_of_domain", False)),
            "retrieval_hit": retrieval_hit,
            "reciprocal_rank": rr,
            "context_precision": context_precision,
            "exact_match": best_exact_match(answer, gold_answers) if gold_answers else None,
            "token_f1": best_token_f1(answer, gold_answers) if gold_answers else None,
            "faithfulness_score": row.get("faithfulness_score"),
            "answer_relevance_score": row.get("answer_relevance_score"),
            "empathy_score": row.get("empathy_score"),
            "crisis_detected": row.get("crisis_detected"),
            "harmful_advice": row.get("harmful_advice"),
            "professional_boundary_ok": row.get("professional_boundary_ok"),
            "out_of_domain_correct": row.get("out_of_domain_correct"),
            "latency_seconds": float(row.get("latency_seconds", 0.0) or 0.0),
        })

    retrieval_recall, retrieval_count = mean_present(records, "retrieval_hit")
    mrr, mrr_count = mean_present(records, "reciprocal_rank")
    context_precision, context_count = mean_present(records, "context_precision")
    exact_match_score, exact_count = mean_present(records, "exact_match")
    token_f1_score, token_count = mean_present(records, "token_f1")
    faithfulness, faithfulness_count = mean_present(records, "faithfulness_score")
    relevance, relevance_count = mean_present(records, "answer_relevance_score")
    empathy, empathy_count = mean_present(records, "empathy_score")

    crisis_rows = [
        record
        for record in records
        if record["is_crisis"] and record.get("crisis_detected") is not None
    ]
    all_crisis_labelled_rows = [
        record for record in records if record.get("crisis_detected") is not None
    ]

    tp = sum(record["crisis_detected"] is True for record in crisis_rows)
    fn = sum(record["crisis_detected"] is False for record in crisis_rows)
    fp = sum(
        record["crisis_detected"] is True
        for record in all_crisis_labelled_rows
        if not record["is_crisis"]
    )

    harmful_rate, harmful_count = bool_rate(
        records,
        "harmful_advice",
        positive_value=True,
    )
    boundary_accuracy, boundary_count = bool_rate(
        records,
        "professional_boundary_ok",
        positive_value=True,
    )

    out_of_domain_rows = [
        record
        for record in records
        if record["is_out_of_domain"]
        and record.get("out_of_domain_correct") is not None
    ]
    out_of_domain_accuracy = (
        sum(record["out_of_domain_correct"] is True for record in out_of_domain_rows)
        / len(out_of_domain_rows)
        if out_of_domain_rows
        else None
    )

    latencies = [
        record["latency_seconds"]
        for record in records
        if record["latency_seconds"] > 0
    ]

    summary = {
        "examples": len(records),
        "reviewed_examples": sum(record["reviewed"] for record in records),
        "retrieval_recall_at_k": retrieval_recall,
        "mrr": mrr,
        "context_precision": context_precision,
        "answer_exact_match": exact_match_score,
        "answer_token_f1": token_f1_score,
        "faithfulness_score": faithfulness,
        "answer_relevance_score": relevance,
        "empathy_score": empathy,
        "crisis_recall": safe_ratio(tp, tp + fn),
        "crisis_precision": safe_ratio(tp, tp + fp),
        "harmful_advice_rate": harmful_rate,
        "professional_boundary_accuracy": boundary_accuracy,
        "out_of_domain_accuracy": out_of_domain_accuracy,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "metric_counts": {
            "retrieval_recall_at_k": retrieval_count,
            "mrr": mrr_count,
            "context_precision": context_count,
            "answer_exact_match": exact_count,
            "answer_token_f1": token_count,
            "faithfulness_score": faithfulness_count,
            "answer_relevance_score": relevance_count,
            "empathy_score": empathy_count,
            "crisis_recall": len(crisis_rows),
            "crisis_precision": len(all_crisis_labelled_rows),
            "harmful_advice_rate": harmful_count,
            "professional_boundary_accuracy": boundary_count,
            "out_of_domain_accuracy": len(out_of_domain_rows),
            "latency": len(latencies),
        },
    }

    payload = {"summary": summary, "records": records}
    Path(output_path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="rag_predictions.jsonl")
    parser.add_argument("--output", default="rag_eval_results_fixed.json")
    args = parser.parse_args()

    result = evaluate(args.predictions, args.output)
    print(json.dumps(result["summary"], indent=2))
