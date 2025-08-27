from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from pathlib import Path

def load_resume(file_path: str):
    """
    Loads resume documents from PDF or DOCX format and returns LangChain Document objects.
    """
    file = Path(file_path)

    if not file.exists():
        raise FileNotFoundError(f" File not found: {file_path}")

    if file.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file))
    elif file.suffix.lower() in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(str(file))
    else:
        raise ValueError(f" Unsupported file type: {file.suffix}")

    documents = loader.load()
    return documents
