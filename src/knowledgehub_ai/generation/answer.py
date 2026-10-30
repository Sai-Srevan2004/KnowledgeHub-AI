
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from knowledgehub_ai.generation.llm import get_chat_model
from knowledgehub_ai.generation.prompt import RAG_PROMPT


def format_context(documents: list[Document]) -> str:
    """
    Convert retrieved documents into context text.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")

        context_parts.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """
    Generate an answer using the question and retrieved context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(documents)

    if not context:
        return "I couldn't find relevant information in the provided documents."

    llm = get_chat_model()

    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer
