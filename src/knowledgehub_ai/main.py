
from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store
from knowledgehub_ai.retrieval.retriever import retrieve_documents
from knowledgehub_ai.generation.answer import generate_answer


def main():
    # 1. Ingest documents.
    chunks = ingest_documents("data/documents")

    # 2. Build vector index.
    print("\nBuilding vector index...")
    vector_store = build_vector_store(chunks)

    # 3. Accept a user question.
    question = input("\nAsk KnowledgeHub AI: ")

    # 4. Retrieve relevant chunks.
    print("\nRetrieving relevant documents...")

    retrieved_documents = retrieve_documents(
        vector_store=vector_store,
        query=question,
        top_k=3,
    )

        # 5. Generate answer using retrieved context.
    print("\nGenerating answer...")

    response = generate_answer(
        question=question,
        documents=retrieved_documents,
    )

    print("\n========== KNOWLEDGEHUB AI ==========")

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")

    for source in response["sources"]:
        print(f"\nReference: {source['reference']}")
        print(f"Document: {source['source']}")
        print(f"Chunk: {source['chunk_index']}")
        print(f"Content: {source['content']}")

if __name__ == "__main__":
    main()
