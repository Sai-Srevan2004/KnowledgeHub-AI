
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_NAME = os.getenv("LLM_NAME")


def validate_config() -> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from .env")

    if not LLM_NAME:
        raise ValueError("LLM_NAME is missing from .env")
