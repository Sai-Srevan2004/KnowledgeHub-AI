from knowledgehub_ai.ingestion.pipeline import ingest_documents
from knowledgehub_ai.indexing.vector_store import build_vector_store
from knowledgehub_ai.retrieval.retriever import retrieve_documents
from knowledgehub_ai.generation.answer import generate_answer


def main():
    # 1. Ingest documents once.
    print("Loading and processing documents...")

    chunks = ingest_documents("data/documents")

    # 2. Build the vector index once.
    print("\nBuilding vector index...")

    vector_store = build_vector_store(chunks)

    print("\nKnowledgeHub AI is ready!")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.")

    # 3. Accept multiple questions.
    while True:
        question = input("\nAsk KnowledgeHub AI: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nClosing KnowledgeHub AI. Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        # 4. Retrieve relevant chunks.
        print("\nRetrieving relevant documents...")

        retrieved_documents = retrieve_documents(
            vector_store=vector_store,
            query=question,
            top_k=3,
        )
        
        # 5. Generate answer with sources.
        print("\nGenerating answer...")

        response = generate_answer(
            question=question,
            documents=retrieved_documents,
        )

        # 6. Display answer.
        print("\n========== KNOWLEDGEHUB AI ==========")

        print("\nAnswer:")
        print(response["answer"])

        # 7. Display sources.
        print("\nSources:")

        for source in response["sources"]:
            print(f"\nReference: {source['reference']}")
            print(f"Document: {source['source']}")
            print(f"Chunk: {source['chunk_index']}")
        


if __name__ == "__main__":
    main()