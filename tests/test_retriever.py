
import pytest

from knowledgehub_ai.retrieval.retriever import retrieve_documents


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_documents(
            vector_store=None,
            query="   ",
        )


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retrieve_documents(
            vector_store=None,
            query="What is parental leave?",
            top_k=0,
        )
