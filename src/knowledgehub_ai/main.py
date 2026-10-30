
from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store


def main():
    # Step 1: Load, clean and split documents.
    chunks = ingest_documents("data/documents")

    # Step 2: Generate embeddings and build the index.
    print("\nBuilding vector index...")
    vector_store = build_vector_store(chunks)

    print("Vector index created successfully!")

    # Step 3: Perform a small smoke test.
    # We will study retrieval properly in Lesson 36.
    results = vector_store.similarity_search(
        "How much parental leave can an employee take?",
        k=1,
    )

    print("\n--- INDEX VERIFICATION ---")

    for result in results:
        print("Retrieved content:")
        print(result.page_content)

        print("\nRetrieved metadata:")
        print(result.metadata)


if __name__ == "__main__":
    main()
