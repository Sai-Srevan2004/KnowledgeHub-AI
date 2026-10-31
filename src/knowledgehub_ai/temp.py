from knowledgehub_ai.ingestion.pipeline import ingest_documents

chunks = ingest_documents("data/documents")

print("\n========== CHUNK DISTRIBUTION ==========")

distribution = {}

for chunk in chunks:
    source = chunk.metadata.get("source", "Unknown")
    distribution[source] = distribution.get(source, 0) + 1

for source, count in distribution.items():
    print(f"{source}: {count} chunks")

print(f"\nTotal chunks: {len(chunks)}")