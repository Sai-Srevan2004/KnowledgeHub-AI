import json
from pathlib import Path

from knowledgehub_ai.retrieval.query_rewriter import rewrite_query


def main():
    input_file = Path("data/evaluation/ground_truth.json")
    output_file = Path("data/evaluation/rewritten_queries.json")

    dataset = json.loads(input_file.read_text(encoding="utf-8"))

    rewritten_queries = {}

    for item in dataset:
        if not item["answerable"]:
            continue

        question_id = item["id"]
        question = item["question"]

        print(f"\nRewriting {question_id}: {question}")

        rewritten = rewrite_query(question)

        rewritten_queries[question_id] = {
            "original_question": question,
            "rewritten_query": rewritten,
        }

        print(f"Rewritten: {rewritten}")

    output_file.write_text(
        json.dumps(
            rewritten_queries,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nGenerated rewrites: {len(rewritten_queries)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()