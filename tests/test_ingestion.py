
from langchain_core.documents import Document

from knowledgehub_ai.ingestion.cleaner import clean_documents
from knowledgehub_ai.ingestion.chunker import split_documents


def test_clean_documents():
    documents = [
        Document(
            page_content="Hello     world\n\n\n\nWelcome",
            metadata={"source": "test.txt"},
        )
    ]

    cleaned = clean_documents(documents)

    assert len(cleaned) == 1
    assert cleaned[0].page_content == "Hello world\n\nWelcome"
    assert cleaned[0].metadata["source"] == "test.txt"


def test_split_documents():
    documents = [
        Document(
            page_content="A " * 500,
            metadata={"source": "test.txt"},
        )
    ]

    chunks = split_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all("chunk_index" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)
