from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store


def main():
    chunks = ingest_documents("data/documents")
    vector_store = build_vector_store(chunks)

    questions = [
        "How should employees report a lost or stolen company laptop?",
        "Where may employees obtain software for company devices?",
        "What details and documents must an employee include when submitting an expense claim?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = vector_store.similarity_search_with_score(
            query=question,
            k=5,
        )

        for rank, (document, score) in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk ID: {document.metadata['chunk_index']}")
            print(f"Source: {document.metadata['source']}")
            print(f"Similarity Score: {score:.4f}")
            print(f"Content: {document.page_content}")


if __name__ == "__main__":
    main()