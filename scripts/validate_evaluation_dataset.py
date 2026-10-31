
import json
from pathlib import Path


def load_json(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    evaluation_directory = Path("data/evaluation")

    inventory = load_json(
        evaluation_directory / "chunk_inventory.json"
    )

    ground_truth = load_json(
        evaluation_directory / "ground_truth.json"
    )

    valid_chunk_ids = {
        chunk["chunk_id"] for chunk in inventory
    }

    errors = []

    required_fields = {
        "id",
        "question",
        "expected_answer",
        "source",
        "relevant_chunk_ids",
        "relevance_grades",
        "question_type",
        "answerable",
    }

    for item in ground_truth:
        missing_fields = required_fields - item.keys()

        if missing_fields:
            errors.append(
                f"{item.get('id', 'Unknown')}: Missing fields {missing_fields}"
            )
            continue

        chunk_ids = item["relevant_chunk_ids"]

        if item["answerable"] and not chunk_ids:
            errors.append(
                f"{item['id']}: Answerable question has no relevant chunks"
            )

        if not item["answerable"] and chunk_ids:
            errors.append(
                f"{item['id']}: Unanswerable question references chunks"
            )

        for chunk_id in chunk_ids:
            if chunk_id not in valid_chunk_ids:
                errors.append(
                    f"{item['id']}: Invalid chunk ID {chunk_id}"
                )

    print("\n========== EVALUATION DATASET VALIDATION ==========")

    print(f"Inventory chunks: {len(inventory)}")
    print(f"Evaluation questions: {len(ground_truth)}")

    if errors:
        print("\nValidation failed:")

        for error in errors:
            print(f"- {error}")

        raise ValueError("Ground-truth dataset contains errors.")

    answerable_count = sum(
        1 for item in ground_truth if item["answerable"]
    )

    unanswerable_count = len(ground_truth) - answerable_count

    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")
    print("\nAll validation checks passed!")


if __name__ == "__main__":
    main()
