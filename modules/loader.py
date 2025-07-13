import os
from typing import List, Type
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

class DocumentLoader:
    """
    General-purpose document loader with support for various formats.
    """

    loaders_map: dict[str, Type] = {
        ".pdf": PyMuPDFLoader,  
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".txt": TextLoader,
    }

    def load(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self.loaders_map:
            raise ValueError(f"❌ Unsupported file type: {ext}")

        loader_class = self.loaders_map[ext]

        try:
            loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            raise RuntimeError(f"🚨 Error loading file: {e}")

    @staticmethod
    def extract_text(documents: List[Document]) -> str:
        """
        Merge contents of all Document objects into a single plain text.
        """
        return "\n\n".join(doc.page_content for doc in documents)
