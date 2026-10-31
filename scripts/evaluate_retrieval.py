
import json
import math
from pathlib import Path

from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store


TOP_K = 5


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_count / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def dcg_at_k(retrieved_ids, relevance_grades, k):
    score = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        relevance = relevance_grades.get(str(chunk_id), 0)

        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(retrieved_ids, relevance_grades, k):
    actual_dcg = dcg_at_k(
        retrieved_ids,
        relevance_grades,
        k,
    )

    ideal_grades = sorted(
        relevance_grades.values(),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        (2 ** relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_grades, start=1)
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def main():
    evaluation_directory = Path("data/evaluation")

    ground_truth_path = evaluation_directory / "ground_truth.json"


    with ground_truth_path.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    evaluation_questions = [
        item for item in ground_truth
        if item["answerable"]
    ]

    print("Loading and processing documents...")

    chunks = ingest_documents("data/documents")

    print("\nBuilding vector index...")

    vector_store = build_vector_store(chunks)

    results = []

    for item in evaluation_questions:
        question = item["question"]

        relevant_ids = set(item["relevant_chunk_ids"])
        relevance_grades = item["relevance_grades"]

        retrieved_documents = vector_store.similarity_search(
            query=question,
            k=TOP_K,
        )

        retrieved_ids = [
            document.metadata["chunk_index"]
            for document in retrieved_documents
        ]

        metrics = {
            "precision": precision_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
            "recall": recall_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
            "mrr": reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "ndcg": ndcg_at_k(
                retrieved_ids, relevance_grades, TOP_K
            ),
        }

        results.append({
            "id": item["id"],
            "question": question,
            "expected_chunk_ids": sorted(relevant_ids),
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": metrics,
        })

        print(
            f"{item['id']} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"MRR: {metrics['mrr']:.3f} | "
            f"nDCG: {metrics['ndcg']:.3f}"
        )

    metric_names = ["precision", "recall", "mrr", "ndcg"]

    averages = {
        metric: sum(
            result["metrics"][metric] for result in results
        ) / len(results)
        for metric in metric_names
    }

    report = {
        "top_k": TOP_K,
        "total_questions": len(results),
        "averages": averages,
        "results": results,
    }

    output_path = evaluation_directory / "retrieval_top5.json"

    output_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== RETRIEVAL BASELINE ==========")

    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nEvaluated questions: {len(results)}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
