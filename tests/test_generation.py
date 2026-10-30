
from langchain_core.documents import Document

from knowledgehub_ai.generation.answer import format_context


def test_format_context():
    documents = [
        Document(
            page_content="Employees receive 16 weeks of parental leave.",
            metadata={"source": "employee_handbook.txt"},
        )
    ]

    context = format_context(documents)

    assert "16 weeks" in context
    assert "employee_handbook.txt" in context


def test_empty_context():
    context = format_context([])

    assert context == ""
