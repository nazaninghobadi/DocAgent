from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder(Embeddings):
    """
    Adapter for HuggingFace SentenceTransformer to be compatible with LangChain Embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode(text, convert_to_numpy=True).tolist()