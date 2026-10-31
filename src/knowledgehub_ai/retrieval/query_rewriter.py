from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from knowledgehub_ai.generation.llm import get_chat_model


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a query rewriting assistant for an employee
            policy knowledge base.

            Rewrite the user's question into a clear search query.

            Rules:
            - Preserve the original meaning.
            - Do not answer the question.
            - Do not introduce new facts.
            - Keep important names and entities.
            - Use terminology likely to appear in policy documents.
            - Return only the rewritten query.
            """,
        ),
        (
            "human",
            "Original question: {question}",
        ),
    ]
)


def rewrite_query(question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    llm = get_chat_model()

    chain = REWRITE_PROMPT | llm | StrOutputParser()

    rewritten_query = chain.invoke(
        {"question": question}
    )

    return rewritten_query.strip()