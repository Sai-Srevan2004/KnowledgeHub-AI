import json
from pathlib import Path

from knowledgehub_ai.ingestion.pipeline import ingest_documents


def main():
    chunks = ingest_documents("data/documents")

    inventory = []

    for index, chunk in enumerate(chunks):
        inventory.append({
            "chunk_id": index,
            "source": chunk.metadata.get("source"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "content": chunk.page_content,
        })

    output_directory = Path("data/evaluation")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "chunk_inventory.json"

    output_file.write_text(
        json.dumps(inventory, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== CHUNK INVENTORY ==========")

    for chunk in inventory:
        preview = chunk["content"][:150].replace("\n", " ")

        print(
            f"\nID: {chunk['chunk_id']}"
            f"\nSource: {chunk['source']}"
            f"\nContent: {preview}..."
        )

    print(f"\nTotal chunks: {len(inventory)}")
    print(f"Inventory saved to: {output_file}")


if __name__ == "__main__":
    main()