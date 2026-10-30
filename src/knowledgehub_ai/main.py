
from knowledgehub_ai.ingestion.pipeline import ingest_documents


def main():
    chunks = ingest_documents("data/documents")

    print("\n--- INGESTION RESULT ---")

    for chunk in chunks:
        print("\nChunk metadata:")
        print(chunk.metadata)

        print("Chunk content:")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
