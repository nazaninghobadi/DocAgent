from typing import Optional, List
import logging
from functools import lru_cache
from langchain_core.tools import tool
from langchain_core.documents import Document

from modules.vector_store import VectorStoreManager
from modules.embeddings import HuggingFaceEmbedder
from modules.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton pattern for managing model instances and caching.
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embedding_model(self) -> HuggingFaceEmbedder:
        """Get cached embedding model."""
        if 'embedding' not in self._models:
            logger.info("Initializing embedding model")
            self._models['embedding'] = HuggingFaceEmbedder()
        return self._models['embedding']
    
    def get_vector_manager(self, save_path: str = "data/faiss_index") -> VectorStoreManager:
        """Get cached vector store manager."""
        cache_key = f"vector_manager_{save_path}"
        if cache_key not in self._models:
            logger.info(f"Initializing vector store manager for {save_path}")
            embedding_model = self.get_embedding_model()
            self._models[cache_key] = VectorStoreManager(
                embedding_model=embedding_model,
                save_path=save_path
            )
        return self._models[cache_key]
    
    def get_llm(self, api_key: str, model_name: str = "mistralai/mistral-7b-instruct") -> LLMProvider:
        """Get cached LLM provider."""
        cache_key = f"llm_{model_name}"
        if cache_key not in self._models:
            logger.info(f"Initializing LLM provider for {model_name}")
            self._models[cache_key] = LLMProvider(api_key=api_key, model_name=model_name)
        return self._models[cache_key]
    
    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()
        logger.info("Model cache cleared")


# Global model manager instance
model_manager = ModelManager()


@tool
def search_knowledge(query: str, k: int = 3, score_threshold: Optional[float] = None) -> str:
    """
    Search documents using vector similarity with advanced filtering.
    
    Args:
        query: Natural language search query
        k: Number of top results to return (default: 3)
        score_threshold: Minimum similarity score filter (optional)
    
    Returns:
        Formatted search results or error message
    """
    if not query.strip():
        return "[Error: Empty query provided]"
    
    try:
        # Get cached vector manager
        vector_manager = model_manager.get_vector_manager()
        
        # Perform search with optional score filtering
        if score_threshold is not None:
            results = vector_manager.search(query, k=k, score_threshold=score_threshold)
        else:
            results = vector_manager.search(query, k=k)
        
        if not results:
            return "[No relevant documents found]"
        
        # Format results with metadata
        formatted_results = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            # Add result number and metadata info
            result_text = f"Result {i}:\n{content}"
            if metadata:
                source = metadata.get('source', 'Unknown')
                result_text += f"\n[Source: {source}]"
            
            formatted_results.append(result_text)
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search knowledge error: {str(e)}")
        return f"[Error: {str(e)}]"


@tool
def search_knowledge_with_scores(query: str, k: int = 3) -> str:
    """
    Search documents and return results with similarity scores.
    
    Args:
        query: Natural language search query
        k: Number of top results to return
    
    Returns:
        Search results with similarity scores
    """
    if not query.strip():
        return "[Error: Empty query provided]"
    
    try:
        vector_manager = model_manager.get_vector_manager()
        results_with_scores = vector_manager.search_with_scores(query, k=k)
        
        if not results_with_scores:
            return "[No relevant documents found]"
        
        formatted_results = []
        for i, (doc, score) in enumerate(results_with_scores, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            
            result_text = f"Result {i} (Score: {score:.3f}):\n{content}\n[Source: {source}]"
            formatted_results.append(result_text)
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search with scores error: {str(e)}")
        return f"[Error: {str(e)}]"


def create_summarize_tool(api_key: str, model_name: str = "mistralai/mistral-7b-instruct"):
    """
    Factory function to create a summarize tool with specific API key.
    
    Args:
        api_key: OpenRouter API key
        model_name: Model name to use for summarization
    
    Returns:
        Configured summarize tool
    """
    
    @tool
    def summarize_knowledge(
        query: str,
        max_docs: int = 5,
        summary_length: str = "medium"
    ) -> str:
        """
        Summarize relevant documents based on query with customizable length.
        
        Args:
            query: Topic or question to summarize about
            max_docs: Maximum number of documents to include (default: 5)
            summary_length: Length of summary - "short", "medium", or "long"
        
        Returns:
            AI-generated summary of relevant documents
        """
        if not query.strip():
            return "[Error: Empty query provided]"
        
        # Validate summary length
        if summary_length not in ["short", "medium", "long"]:
            summary_length = "medium"
        
        try:
            # Get relevant documents
            vector_manager = model_manager.get_vector_manager()
            docs = vector_manager.search(query, k=max_docs)
            
            if not docs:
                return "[No relevant documents found for summarization]"
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', f'Document {i}')
                context_parts.append(f"Document {i} ({source}):\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Create length-specific prompts
            length_prompts = {
                "short": "Write a concise summary in 2-3 sentences",
                "medium": "Write a comprehensive summary in 1-2 paragraphs",
                "long": "Write a detailed summary with key points and examples"
            }
            
            prompt = f"""
{length_prompts[summary_length]} of the following documents related to: {query}

Context:
{context}

Summary:"""
            
            # Generate summary
            llm_provider = model_manager.get_llm(api_key, model_name)
            llm = llm_provider.get_chat_model()
            
            response = llm.invoke(prompt)
            summary = response.content.strip()
            
            # Add metadata
            metadata_info = f"\n\n[Summary based on {len(docs)} documents, Length: {summary_length}]"
            
            return summary + metadata_info
            
        except Exception as e:
            logger.error(f"Summarize knowledge error: {str(e)}")
            return f"[Error: {str(e)}]"
    
    return summarize_knowledge


@tool
def get_vector_store_stats() -> str:
    """
    Get statistics about the current vector store.
    
    Returns:
        Formatted statistics about the vector store
    """
    try:
        vector_manager = model_manager.get_vector_manager()
        stats = vector_manager.get_stats()
        
        return f"""
Vector Store Statistics:
- Total Documents: {stats['total_documents']}
- Embedding Dimension: {stats['embedding_dimension']}
- Is Loaded: {stats['is_loaded']}
- Save Path: {stats['save_path']}
- Cache Enabled: {stats['cache_enabled']}
"""
    
    except Exception as e:
        logger.error(f"Get stats error: {str(e)}")
        return f"[Error: {str(e)}]"


# Helper function for backward compatibility
def get_summarize_tool(api_key: str):
    """
    Backward compatibility function.
    
    Args:
        api_key: OpenRouter API key
    
    Returns:
        Configured summarize tool
    """
    return create_summarize_tool(api_key)
