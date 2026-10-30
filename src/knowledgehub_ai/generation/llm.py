
from langchain_groq import ChatGroq

from knowledgehub_ai.config import (
    GROQ_API_KEY,
    LLM_NAME,
    validate_config,
)


def get_chat_model() -> ChatGroq:
    """
    Initialize the Groq chat model.
    """

    validate_config()

    return ChatGroq(
        model=LLM_NAME,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
