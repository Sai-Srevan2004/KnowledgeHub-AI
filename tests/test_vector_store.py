
import pytest

from knowledgehub_ai.indexing.vector_store import build_vector_store


def test_empty_chunks_raise_error():
    with pytest.raises(ValueError, match="no document chunks"):
        build_vector_store([])
