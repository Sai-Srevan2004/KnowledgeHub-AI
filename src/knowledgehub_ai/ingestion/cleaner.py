
import re

from langchain_core.documents import Document


def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned_documents = []

    for document in documents:
        cleaned_text = document.page_content

        # Replace multiple spaces and tabs with one space.
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        # Replace excessive blank lines with one blank line.
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        cleaned_text = cleaned_text.strip()

        if not cleaned_text:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=document.metadata.copy(),
            )
        )

    return cleaned_documents
