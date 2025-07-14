# import os
# from typing import List
# from langchain_core.documents import Document
# from langchain.vectorstores import FAISS
# from langchain.embeddings.base import Embeddings


# class VectorStoreManager:
#     def __init__(self, embedding_model: Embeddings, save_path: str = "faiss_index"):
#         self.embedding_model = embedding_model
#         self.save_path = save_path
#         self.vector_store = None

#     def create_vector_store(self, documents: List[Document]):
#         """ساخت پایگاه برداری از اسناد"""
#         self.vector_store = FAISS.from_documents(documents, self.embedding_model)
#         self.save_vector_store()

#     def save_vector_store(self):
#         """ذخیره پایگاه برداری در دیسک"""
#         if self.vector_store:
#             self.vector_store.save_local(self.save_path)

#     def load_vector_store(self):
#         """بارگذاری پایگاه برداری از دیسک"""
#         if os.path.exists(self.save_path):
#             self.vector_store = FAISS.load_local(self.save_path, self.embedding_model)
#         else:
#             raise FileNotFoundError(f"Vector store not found at: {self.save_path}")

#     def similarity_search(self, query: str, k: int = 3) -> List[Document]:
#         """انجام سرچ در بردارها"""
#         if self.vector_store is None:
#             self.load_vector_store()
#         return self.vector_store.similarity_search(query, k=k)





from typing import List
import os

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    """
    Manages a FAISS-based vector store using a given embedding model.
    """
    def __init__(self, embedding_model, save_path: str = "data/faiss_index"):
        """
    Args:
        embedding_model: An object implementing LangChain's Embeddings interface.
        save_path (str): Path to save/load FAISS index.
    """
    self.embedding_model = embedding_model
    self.save_path = save_path
    self.vector_store = None

def create_vector_store(self, documents: List[Document]) -> None:
    """Creates a FAISS index from documents and saves it."""
    self.vector_store = FAISS.from_documents(documents, self.embedding_model)
    self.save_vector_store()

def save_vector_store(self) -> None:
    """Saves FAISS index to disk."""
    if self.vector_store:
        self.vector_store.save_local(self.save_path)
    else:
        raise ValueError("Vector store not created yet.")

def load_vector_store(self) -> None:
    """Loads FAISS index from disk."""
    if os.path.exists(self.save_path):
        self.vector_store = FAISS.load_local(self.save_path, self.embedding_model)
    else:
        raise FileNotFoundError(f"No FAISS index found at: {self.save_path}")

def search(self, query: str, k: int = 3) -> List[Document]:
    """Performs similarity search on the vector store."""
    if not self.vector_store:
        raise ValueError("Vector store not loaded or created.")
    return self.vector_store.similarity_search(query, k=k)



