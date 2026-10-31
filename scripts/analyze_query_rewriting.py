import json
from pathlib import Path


def main():
    evaluation_dir = Path("data/evaluation")

    baseline = json.loads(
        (evaluation_dir / "retrieval_baseline.json").read_text(
            encoding="utf-8"
        )
    )

    rewritten = json.loads(
        (evaluation_dir / "retrieval_query_rewriting.json").read_text(
            encoding="utf-8"
        )
    )

    baseline_results = {
        item["id"]: item
        for item in baseline["results"]
    }

    rewritten_results = {
        item["id"]: item
        for item in rewritten["question_results"]
    }

    print("\n========== QUERY REWRITING REGRESSION ANALYSIS ==========")

    regression_count = 0

    for question_id, new_result in rewritten_results.items():
        old_result = baseline_results[question_id]

        old_retrieved = old_result["retrieved_chunk_ids"]
        new_retrieved = new_result["retrieved_chunks"]

        old_relevant = old_result["expected_chunk_ids"]
        new_relevant = new_result["relevant_chunks"]

        old_hits = len(set(old_retrieved) & set(old_relevant))
        new_hits = len(set(new_retrieved) & set(new_relevant))

        if new_hits < old_hits:
            regression_count += 1

            print(f"\n{question_id}")
            print(f"Question: {new_result['original_question']}")
            print(f"Rewritten: {new_result['rewritten_query']}")
            print(f"Baseline retrieved: {old_retrieved}")
            print(f"Rewritten retrieved: {new_retrieved}")
            print(f"Relevant chunks: {old_relevant}")
            print(f"Baseline hits: {old_hits}")
            print(f"Rewritten hits: {new_hits}")

    print(f"\nTotal questions with fewer relevant chunks: {regression_count}")


if __name__ == "__main__":
    main()