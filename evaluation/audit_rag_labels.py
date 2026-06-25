"""Audit label coverage in rag_predictions.jsonl before evaluation."""

import argparse
import json
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(path: str) -> None:
    rows = load_jsonl(path)

    fields = [
        "reviewed",
        "context_relevant_doc_ids",
        "gold_relevant_doc_ids",
        "relevant_doc_ids",
        "gold_answers",
        "faithfulness_score",
        "answer_relevance_score",
        "empathy_score",
        "crisis_detected",
        "harmful_advice",
        "professional_boundary_ok",
        "out_of_domain_correct",
    ]

    print(f"Total records: {len(rows)}\n")

    for field in fields:
        present = sum(
            field in row and row[field] is not None
            for row in rows
        )
        non_empty = sum(bool(row.get(field)) for row in rows)

        print(
            f"{field:<32} "
            f"present={present:<3} "
            f"non_empty/true={non_empty:<3}"
        )

    print("\nCategory counts:")
    print(
        dict(
            Counter(
                row.get("category", "uncategorized")
                for row in rows
            )
        )
    )

    print("\nPer-record review status:")

    for row in rows:
        print(
            f"{row.get('id', '?'):<10} "
            f"reviewed={row.get('reviewed')} "
            f"faith={row.get('faithfulness_score')} "
            f"relevance={row.get('answer_relevance_score')} "
            f"empathy={row.get('empathy_score')} "
            f"crisis={row.get('is_crisis')} "
            f"detected={row.get('crisis_detected')} "
            f"boundary={row.get('professional_boundary_ok')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="rag_predictions.jsonl",
    )
    args = parser.parse_args()

    main(args.predictions)
