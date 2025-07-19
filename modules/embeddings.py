from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import re

logger = logging.getLogger(__name__)


class AdvancedHuggingFaceEmbedder(Embeddings):
    """
    Advanced HuggingFace SentenceTransformer adapter with preprocessing and optimization.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_seq_length: Optional[int] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the advanced embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
            cache_folder: Folder to cache models
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize model
        self._initialize_model(max_seq_length)
        
        # Text preprocessing patterns
        self._setup_preprocessing()
        
        logger.info(f"Initialized embedder: {model_name} on {device}")

    def _initialize_model(self, max_seq_length: Optional[int]) -> None:
        """Initialize the SentenceTransformer model with optimizations."""
        try:
            model_kwargs = {
                'device': self.device,
                'cache_folder': self.cache_folder
            }
            
            self.model = SentenceTransformer(self.model_name, **model_kwargs)
            
            # Set maximum sequence length if specified
            if max_seq_length:
                self.model.max_seq_length = max_seq_length
            
            # Enable model optimization for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _setup_preprocessing(self) -> None:
        """Setup text preprocessing patterns and rules."""
        # Common patterns to clean
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple whitespaces
            (r'\n+', '\n'),  # Multiple newlines
            (r'[^\w\s\.\,\!\?\:\;\-\(\)]', ''),  # Special characters (keep basic punctuation)
            (r'\.{2,}', '...'),  # Multiple dots
            (r'\?{2,}', '?'),  # Multiple question marks
            (r'\!{2,}', '!'),  # Multiple exclamation marks
        ]
        
        # Academic/technical text patterns
        self.academic_patterns = [
            (r'\b(Fig\.|Figure|Table|Eq\.|Equation)\s*\d+', '[FIGURE_REF]'),
            (r'\[\d+\]', '[CITATION]'),
            (r'\b(et\s+al\.)', '[ET_AL]'),
            (r'\b(i\.e\.|e\.g\.)', '[ABBREV]'),
        ]

    def _preprocess_text(self, text: str, preserve_academic: bool = True) -> str:
        """
        Advanced text preprocessing for better embeddings.
        
        Args:
            text: Input text
            preserve_academic: Whether to preserve academic patterns
            
        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        # Basic cleaning
        processed = text.strip()
        
        # Apply academic patterns if requested
        if preserve_academic:
            for pattern, replacement in self.academic_patterns:
                processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            processed = re.sub(pattern, replacement, processed)
        
        # Additional cleaning
        processed = processed.strip()
        
        # Ensure minimum length
        if len(processed) < 10:
            logger.warning(f"Very short text after preprocessing: '{processed[:50]}...'")
        
        return processed

    def _batch_encode(
        self,
        texts: List[str],
        show_progress: bool = False,
        convert_to_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts in batches with optimization.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            convert_to_tensor: Whether to return tensor instead of numpy
            
        Returns:
            Embeddings as numpy array or tensor
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=not convert_to_tensor,
                normalize_embeddings=self.normalize_embeddings,
                device=self.device
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with preprocessing and optimization.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings as float lists
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents")
        
        # Preprocess texts
        processed_texts = []
        for i, text in enumerate(texts):
            try:
                processed = self._preprocess_text(text, preserve_academic=True)
                if not processed:
                    logger.warning(f"Empty text after preprocessing at index {i}")
                    processed = f"[EMPTY_DOCUMENT_{i}]"  # Fallback
                processed_texts.append(processed)
            except Exception as e:
                logger.error(f"Preprocessing failed for document {i}: {str(e)}")
                processed_texts.append(f"[ERROR_DOCUMENT_{i}]")
        
        # Generate embeddings
        embeddings = self._batch_encode(
            processed_texts,
            show_progress=len(texts) > 10
        )
        
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query with optimization for search.
        
        Args:
            text: Query text
            
        Returns:
            Embedding as float list
        """
        if not text or not text.strip():
            logger.warning("Empty query provided")
            return [0.0] * self.get_embedding_dimension()
        
        # Preprocess query (less aggressive than documents)
        processed = self._preprocess_text(text, preserve_academic=False)
        
        # Generate embedding
        embedding = self._batch_encode([processed])
        
        return embedding[0].tolist()

    def embed_query_batch(self, queries: List[str]) -> List[List[float]]:
        """
        Embed multiple queries efficiently.
        
        Args:
            queries: List of query texts
            
        Returns:
            List of embeddings
        """
        if not queries:
            return []
        
        # Preprocess queries
        processed_queries = [
            self._preprocess_text(query, preserve_academic=False) 
            for query in queries
        ]
        
        # Generate embeddings
        embeddings = self._batch_encode(processed_queries)
        
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_seq_length': getattr(self.model, 'max_seq_length', None),
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size
        }

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity score
        """
        embeddings = self._batch_encode([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        if method == 'cosine':
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2)
        elif method == 'dot':
            return np.dot(emb1, emb2)
        elif method == 'euclidean':
            return -np.linalg.norm(emb1 - emb2)  # Negative for similarity
        else:
            raise ValueError(f"Unknown similarity method: {method}")


class SmartEmbeddingGenerator:
    """
    Smart embedding generator with document analysis and optimization.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the smart embedding generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            cache_folder: Cache folder for models
        """
        self.embedder = AdvancedHuggingFaceEmbedder(
            model_name=model_name,
            device=device,
            cache_folder=cache_folder
        )
        
        self.stats = {
            'total_documents': 0,
            'avg_document_length': 0,
            'empty_documents': 0,
            'processing_errors': 0
        }

    def analyze_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze documents to optimize embedding parameters.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Analysis results
        """
        if not documents:
            return {'error': 'No documents provided'}
        
        analysis = {
            'total_documents': len(documents),
            'document_lengths': [],
            'content_types': {'academic': 0, 'general': 0, 'structured': 0},
            'empty_documents': 0,
            'language_patterns': {},
            'recommendations': []
        }
        
        for doc in documents:
            content = doc.page_content
            content_length = len(content)
            analysis['document_lengths'].append(content_length)
            
            if not content.strip():
                analysis['empty_documents'] += 1
                continue
            
            # Detect content type
            if self._is_academic_content(content):
                analysis['content_types']['academic'] += 1
            elif self._is_structured_content(content):
                analysis['content_types']['structured'] += 1
            else:
                analysis['content_types']['general'] += 1
        
        # Calculate statistics
        lengths = analysis['document_lengths']
        if lengths:
            analysis['avg_length'] = sum(lengths) / len(lengths)
            analysis['min_length'] = min(lengths)
            analysis['max_length'] = max(lengths)
            analysis['std_length'] = np.std(lengths)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis

    def _is_academic_content(self, content: str) -> bool:
        """Check if content appears to be academic."""
        academic_indicators = [
            r'\babstract\b', r'\bintroduction\b', r'\bmethodology\b',
            r'\bresults\b', r'\bconclusion\b', r'\breferences\b',
            r'\bcitation\b', r'\bfigure\b', r'\btable\b'
        ]
        
        score = sum(1 for pattern in academic_indicators
                   if re.search(pattern, content, re.IGNORECASE))
        
        return score >= 2

    def _is_structured_content(self, content: str) -> bool:
        """Check if content is structured (headers, lists, etc.)."""
        structure_indicators = [
            r'^#+\s',  # Markdown headers
            r'^\d+\.\s',  # Numbered lists
            r'^[-*]\s',  # Bullet points
            r'^[A-Z][A-Z\s]+:',  # Section headers
        ]
        
        score = sum(1 for pattern in structure_indicators
                   if re.search(pattern, content, re.MULTILINE))
        
        return score >= 2

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on document analysis."""
        recommendations = []
        
        # Check for empty documents
        if analysis['empty_documents'] > 0:
            recommendations.append(
                f"Found {analysis['empty_documents']} empty documents. Consider filtering them out."
            )
        
        # Check document length distribution
        if 'avg_length' in analysis:
            avg_len = analysis['avg_length']
            if avg_len < 100:
                recommendations.append("Documents are very short. Consider merging related chunks.")
            elif avg_len > 2000:
                recommendations.append("Documents are very long. Consider using smaller chunk sizes.")
        
        # Check content type distribution
        content_types = analysis['content_types']
        if content_types['academic'] > content_types['general']:
            recommendations.append("Predominantly academic content detected. Consider using academic-optimized preprocessing.")
        
        if content_types['structured'] > 0:
            recommendations.append("Structured content detected. Consider preserving structure in preprocessing.")
        
        return recommendations

    def generate_embeddings_with_analysis(
        self,
        documents: List[Document],
        analyze_first: bool = True
    ) -> tuple[FAISS, Dict[str, Any]]:
        """
        Generate embeddings with document analysis.
        
        Args:
            documents: List of documents
            analyze_first: Whether to analyze documents first
            
        Returns:
            Tuple of (FAISS vector store, analysis results)
        """
        analysis = {}
        
        if analyze_first:
            logger.info("Analyzing documents before embedding generation...")
            analysis = self.analyze_documents(documents)
            
            # Log recommendations
            if analysis.get('recommendations'):
                for rec in analysis['recommendations']:
                    logger.info(f"Recommendation: {rec}")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        try:
            # Create FAISS vector store
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedder
            )
            
            # Update statistics
            self.stats['total_documents'] += len(documents)
            if 'avg_length' in analysis:
                self.stats['avg_document_length'] = analysis['avg_length']
            self.stats['empty_documents'] += analysis.get('empty_documents', 0)
            
            logger.info("Embedding generation completed successfully")
            
            return vector_store, analysis
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            self.stats['processing_errors'] += 1
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def generate_simple(self, documents: List[Document]) -> FAISS:
        """
        Simple embedding generation without analysis (backward compatibility).
        
        Args:
            documents: List of documents
            
        Returns:
            FAISS vector store
        """
        vector_store, _ = self.generate_embeddings_with_analysis(
            documents, 
            analyze_first=False
        )
        return vector_store

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        return self.stats.copy()

    def benchmark_embedding_speed(self, sample_texts: List[str]) -> Dict[str, float]:
        """
        Benchmark embedding generation speed.
        
        Args:
            sample_texts: Sample texts to benchmark
            
        Returns:
            Benchmark results
        """
        import time
        
        if not sample_texts:
            return {'error': 'No sample texts provided'}
        
        logger.info(f"Benchmarking with {len(sample_texts)} samples...")
        
        # Single document embedding
        start_time = time.time()
        self.embedder.embed_query(sample_texts[0])
        single_time = time.time() - start_time
        
        # Batch embedding
        start_time = time.time()
        self.embedder.embed_documents(sample_texts)
        batch_time = time.time() - start_time
        
        results = {
            'single_embedding_time': single_time,
            'batch_embedding_time': batch_time,
            'documents_per_second': len(sample_texts) / batch_time,
            'speedup_factor': (single_time * len(sample_texts)) / batch_time
        }
        
        logger.info(f"Benchmark results: {results['documents_per_second']:.2f} docs/sec")
        
        return results


# Factory functions for easy initialization
def create_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    **kwargs
) -> AdvancedHuggingFaceEmbedder:
    """Create an advanced embedder with custom parameters."""
    return AdvancedHuggingFaceEmbedder(
        model_name=model_name,
        device=device,
        **kwargs
    )


def create_embedding_generator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
) -> SmartEmbeddingGenerator:
    """Create a smart embedding generator."""
    return SmartEmbeddingGenerator(model_name=model_name, **kwargs)


# Backward compatibility
class EmbeddingGenerator(SmartEmbeddingGenerator):
    """Backward compatibility class."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name)
        # Keep the old embedder reference
        self.embedder = self.embedder

    def generate(self, documents: List[Document]) -> FAISS:
        """Generate embeddings (original interface)."""
        return self.generate_simple(documents)


# Backward compatibility - keep the old HuggingFaceEmbedder
class HuggingFaceEmbedder(AdvancedHuggingFaceEmbedder):
    """Backward compatibility class."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name)