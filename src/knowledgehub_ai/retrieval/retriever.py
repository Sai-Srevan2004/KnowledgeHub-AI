
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def retrieve_documents(
    vector_store: InMemoryVectorStore,
    query: str,
    top_k: int = 3,
) -> list[Document]:
    """
    Retrieve the top-k most similar document chunks.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results = vector_store.similarity_search(
        query=query,
        k=top_k,
    )

    return results


def retrieve_documents_with_scores(
    vector_store: InMemoryVectorStore,
    query: str,
    top_k: int = 3,
) -> list[tuple[Document, float]]:
    """
    Retrieve document chunks along with their similarity scores.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results
