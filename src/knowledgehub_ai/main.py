
from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store
from knowledgehub_ai.retrieval.retriever import (
    retrieve_documents,
    retrieve_documents_with_scores,
)


def main():
    # Step 1: Prepare document chunks.
    chunks = ingest_documents("data/documents")

    # Step 2: Build the vector index.
    print("\nBuilding vector index...")
    vector_store = build_vector_store(chunks)

    # Step 3: Ask a question.
    query = "How many weeks of parental leave can employees take?"

    # Step 4: Retrieve relevant documents.
    print("\n--- RETRIEVED DOCUMENTS ---")

    results = retrieve_documents(
        vector_store=vector_store,
        query=query,
        top_k=3,
    )

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    # Step 5: Inspect similarity scores.
    print("\n--- RETRIEVAL SCORES ---")

    scored_results = retrieve_documents_with_scores(
        vector_store=vector_store,
        query=query,
        top_k=3,
    )

    for document, score in scored_results:
        print(f"\nScore: {score:.4f}")
        print("Source:", document.metadata.get("source"))
        print("Content:", document.page_content)


if __name__ == "__main__":
    main()
