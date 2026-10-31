import json
from pathlib import Path

from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store


TOP_K = 3


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_count = sum(chunk_id in relevant for chunk_id in retrieved_k)

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_count = sum(chunk_id in relevant for chunk_id in retrieved_k)

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved, relevance_grades, k):
    import math

    def dcg(scores):
        return sum(
            (2 ** score - 1) / math.log2(index + 2)
            for index, score in enumerate(scores)
        )

    actual_scores = [
        relevance_grades.get(str(chunk_id), 0)
        for chunk_id in retrieved[:k]
    ]

    ideal_scores = sorted(
        relevance_grades.values(),
        reverse=True,
    )[:k]

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return dcg(actual_scores) / ideal_dcg


def main():
    evaluation_dir = Path("data/evaluation")

    ground_truth = json.loads(
        (evaluation_dir / "ground_truth.json").read_text(
            encoding="utf-8"
        )
    )

    rewritten_queries = json.loads(
        (evaluation_dir / "rewritten_queries.json").read_text(
            encoding="utf-8"
        )
    )

    print("Loading documents and building vector store...")

    chunks = ingest_documents("data/documents")
    vector_store = build_vector_store(chunks)

    results = []

    for item in ground_truth:
        if not item["answerable"]:
            continue

        question_id = item["id"]
        rewritten_query = rewritten_queries[question_id]["rewritten_query"]

        relevant_chunks = item["relevant_chunk_ids"]
        relevance_grades = item["relevance_grades"]

        retrieved_documents = vector_store.similarity_search(
            query=rewritten_query,
            k=TOP_K,
        )

        retrieved_chunk_ids = [
            document.metadata["chunk_index"]
            for document in retrieved_documents
        ]

        precision = precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        recall = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        mrr = reciprocal_rank(
            retrieved_chunk_ids,
            relevant_chunks,
        )

        ndcg = ndcg_at_k(
            retrieved_chunk_ids,
            relevance_grades,
            TOP_K,
        )

        results.append({
            "id": question_id,
            "original_question": item["question"],
            "rewritten_query": rewritten_query,
            "relevant_chunks": relevant_chunks,
            "retrieved_chunks": retrieved_chunk_ids,
            "precision_at_3": precision,
            "recall_at_3": recall,
            "mrr": mrr,
            "ndcg_at_3": ndcg,
        })

    total = len(results)

    summary = {
        "experiment": "Query Rewriting",
        "top_k": TOP_K,
        "total_questions": total,
        "averages": {
            "precision_at_3": sum(
                result["precision_at_3"] for result in results
            ) / total,
            "recall_at_3": sum(
                result["recall_at_3"] for result in results
            ) / total,
            "mrr": sum(
                result["mrr"] for result in results
            ) / total,
            "ndcg_at_3": sum(
                result["ndcg_at_3"] for result in results
            ) / total,
        },
        "question_results": results,
    }

    output_file = evaluation_dir / "retrieval_query_rewriting.json"

    output_file.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== QUERY REWRITING RESULTS ==========")
    print(f"Questions evaluated: {total}")

    for metric, value in summary["averages"].items():
        print(f"{metric}: {value:.4f}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()