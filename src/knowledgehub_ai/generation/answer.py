from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from knowledgehub_ai.generation.llm import get_chat_model
from knowledgehub_ai.generation.prompt import RAG_PROMPT


def format_context(documents: list[Document]) -> str:
    context_parts = []

    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Source {index}]\n{document.page_content}"
        )

    return "\n\n".join(context_parts)


def format_sources(documents: list[Document]) -> list[dict]:
    sources = []

    for index, document in enumerate(documents, start=1):
        sources.append({
            "reference": f"Source {index}",
            "source": document.metadata.get("source", "Unknown source"),
            "chunk_index": document.metadata.get("chunk_index"),
            "content": document.page_content,
        })

    return sources


def generate_answer(
    question: str,
    documents: list[Document],
) -> dict:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(documents)

    # print("==================================\n\n context:",context,"\n\n===============")

    if not context:
        return {
            "answer": "I couldn't find relevant information in the provided documents.",
            "sources": [],
        }

    sources = format_sources(documents)

    llm = get_chat_model()

    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": answer,
        "sources": sources,
    }