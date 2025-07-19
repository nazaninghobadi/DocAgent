from typing import List, Optional, Tuple
import os
import pickle
from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Professional FAISS vector store manager with caching and optimization.
    """
    
    def __init__(
        self,
        embedding_model: Embeddings,
        save_path: str = "data/faiss_index",
        cache_embeddings: bool = True
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: The embedding model to use
            save_path: Path to save/load the vector store
            cache_embeddings: Whether to cache embeddings for performance
        """
        self.embedding_model = embedding_model
        self.save_path = Path(save_path)
        self.cache_embeddings = cache_embeddings
        self.vector_store: Optional[FAISS] = None
        self._is_loaded = False
        self._metadata_cache = {}
        
        # Create directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a FAISS index from documents with optimizations.
        
        Args:
            documents: List of Document objects to index
            
        Raises:
            ValueError: If no documents provided
            RuntimeError: If vector store creation fails
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            # Create FAISS index
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            
            # Cache metadata
            self._cache_metadata(documents)
            
            # Save to disk
            self.save_vector_store()
            self._is_loaded = True
            
            logger.info("Vector store created and saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise RuntimeError(f"Vector store creation failed: {str(e)}")
    
    def save_vector_store(self) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Raises:
            ValueError: If vector store not created
            RuntimeError: If saving fails
        """
        if not self.vector_store:
            raise ValueError("Vector store not created yet")
        
        try:
            # Save FAISS index
            self.vector_store.save_local(str(self.save_path))
            
            # Save metadata cache
            if self.cache_embeddings:
                metadata_path = self.save_path / "metadata_cache.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self._metadata_cache, f)
            
            logger.info(f"Vector store saved to {self.save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise RuntimeError(f"Vector store save failed: {str(e)}")
    
    def load_vector_store(self) -> None:
        """
        Load FAISS index from disk with error handling.
        
        Raises:
            FileNotFoundError: If vector store files not found
            RuntimeError: If loading fails
        """
        if self._is_loaded and self.vector_store:
            logger.debug("Vector store already loaded")
            return
        
        if not self.save_path.exists():
            raise FileNotFoundError(f"No FAISS index found at: {self.save_path}")
        
        try:
            logger.info(f"Loading vector store from {self.save_path}")
            
            # Load FAISS index
            self.vector_store = FAISS.load_local(
                folder_path=str(self.save_path),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata cache
            if self.cache_embeddings:
                metadata_path = self.save_path / "metadata_cache.pkl"
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        self._metadata_cache = pickle.load(f)
            
            self._is_loaded = True
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise RuntimeError(f"Vector store loading failed: {str(e)}")
    
    def search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search with optional score filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of similar documents
        """
        self._ensure_loaded()
        
        if score_threshold is not None:
            # Use similarity search with score
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            return [doc for doc, score in docs_with_scores if score >= score_threshold]
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def search_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        self._ensure_loaded()
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of new documents to add
        """
        self._ensure_loaded()
        
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        try:
            self.vector_store.add_documents(documents)
            self._cache_metadata(documents)
            self.save_vector_store()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RuntimeError(f"Adding documents failed: {str(e)}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing vector store statistics
        """
        self._ensure_loaded()
        
        total_docs = self.vector_store.index.ntotal if self.vector_store else 0
        
        return {
            'total_documents': total_docs,
            'embedding_dimension': self.vector_store.index.d if self.vector_store else 0,
            'is_loaded': self._is_loaded,
            'save_path': str(self.save_path),
            'cache_enabled': self.cache_embeddings
        }
    
    def _ensure_loaded(self) -> None:
        """Ensure vector store is loaded before operations."""
        if not self._is_loaded or not self.vector_store:
            self.load_vector_store()
    
    def _cache_metadata(self, documents: List[Document]) -> None:
        """Cache document metadata for performance."""
        if not self.cache_embeddings:
            return
        
        for doc in documents:
            doc_id = hash(doc.page_content)
            self._metadata_cache[doc_id] = doc.metadata
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._metadata_cache.clear()
        self._is_loaded = False
        self.vector_store = None
        logger.info("Vector store cache cleared")
