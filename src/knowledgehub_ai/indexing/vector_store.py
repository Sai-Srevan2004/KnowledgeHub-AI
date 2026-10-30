
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from knowledgehub_ai.indexing.embeddings import get_embedding_model


def build_vector_store(
    chunks: list[Document],
) -> InMemoryVectorStore:
    """
    Generate embeddings for chunks and store them
    in an in-memory vector index.
    """

    if not chunks:
        raise ValueError("Cannot build an index with no document chunks.")

    embeddings = get_embedding_model()

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store
