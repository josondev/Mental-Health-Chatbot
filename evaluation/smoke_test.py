import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FOLDER = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_FOLDER))

from app import ask_question_for_evaluation


def main():
    question = (
        "What are some grounding techniques "
        "that may help with anxiety?"
    )

    result = ask_question_for_evaluation(question)

    print("\nQUESTION")
    print(question)

    print("\nANSWER")
    print(result["answer"])

    print("\nRETRIEVED DOCUMENTS")
    print("Count:", len(result["documents"]))

    for index, document in enumerate(
        result["documents"],
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Document {index}")
        print("ID:", document["doc_id"])
        print("Metadata:", document["metadata"])
        print("Text:", document["text"][:500])


if __name__ == "__main__":
    main()