import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FOLDER = PROJECT_ROOT / "src"
EVALUATION_FOLDER = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_FOLDER))

from app import ask_question_for_evaluation


TEST_FILE = (
    EVALUATION_FOLDER
    / "mental_health_tests.jsonl"
)

OUTPUT_FILE = (
    EVALUATION_FOLDER
    / "rag_predictions.jsonl"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def run():
    test_cases = load_jsonl(TEST_FILE)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, test_case in enumerate(
            test_cases,
            start=1,
        ):
            print(
                f"[{index}/{len(test_cases)}] "
                f"{test_case['question'][:70]}"
            )

            start_time = time.perf_counter()

            try:
                result = ask_question_for_evaluation(
                    test_case["question"]
                )

                latency = (
                    time.perf_counter()
                    - start_time
                )

                record = {
                    **test_case,
                    "answer": result["answer"],
                    "retrieved_doc_ids": (
                        result["retrieved_doc_ids"]
                    ),
                    "retrieved_documents": (
                        result["documents"]
                    ),
                    "latency_seconds": round(
                        latency,
                        4,
                    ),

                    # Fill these manually later.
                    "faithfulness_score": None,
                    "answer_relevance_score": None,
                    "empathy_score": None,
                    "crisis_detected": None,
                    "harmful_advice": None,
                    "professional_boundary_ok": None,
                    "out_of_domain_correct": None,

                    "error": None,
                }

            except Exception as error:
                record = {
                    **test_case,
                    "answer": "",
                    "retrieved_doc_ids": [],
                    "retrieved_documents": [],
                    "latency_seconds": round(
                        time.perf_counter()
                        - start_time,
                        4,
                    ),
                    "faithfulness_score": 0,
                    "answer_relevance_score": 0,
                    "empathy_score": 0,
                    "crisis_detected": False,
                    "harmful_advice": False,
                    "professional_boundary_ok": False,
                    "out_of_domain_correct": False,
                    "error": str(error),
                }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\nPredictions saved to:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    run()