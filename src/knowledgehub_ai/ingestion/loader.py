
from pathlib import Path

from langchain_core.documents import Document


def load_text_documents(directory: str) -> list[Document]:
    """
    Load all .txt files from a directory.

    Each file becomes a LangChain Document containing:
    - page_content: extracted text
    - metadata: information about the source
    """

    documents = []
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory_path}"
        )

    for file_path in directory_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
            },
        )

        documents.append(document)

    return documents
