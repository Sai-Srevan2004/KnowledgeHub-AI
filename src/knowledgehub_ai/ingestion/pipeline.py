
from langchain_core.documents import Document

from knowledgehub_ai.ingestion.loader import load_text_documents
from knowledgehub_ai.ingestion.cleaner import clean_documents
from knowledgehub_ai.ingestion.chunker import split_documents


def ingest_documents(directory: str) -> list[Document]:

    print("Loading documents...\n")

    documents = load_text_documents(directory)

    print(f"Loaded documents length: {len(documents)}")

    print("Loaded documents:",documents)

    print("\n================================\n")

    print("Cleaning documents...\n")
    cleaned_documents = clean_documents(documents)
    print("cleaned documents:",cleaned_documents)

    print("\n================================\n")
    

    print("Splitting documents into chunks...\n")
    chunks = split_documents(cleaned_documents)

    print(f"Generated chunks: {len(chunks)}")
    print("\nAll chunks:",chunks)

    return chunks
