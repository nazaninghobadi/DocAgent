from typing import List, Optional, Dict, Any
import re
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    length_function: callable = len
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    is_separator_regex: bool = False
    strip_whitespace: bool = True


class DocumentChunker:
    """
    Professional document chunker with multiple strategies and smart splitting.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize document chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'average_chunk_size': 0,
            'min_chunk_size': float('inf'),
            'max_chunk_size': 0
        }
        
        # Initialize text splitters
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize different text splitters."""
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function,
                separators=self.config.separators,
                keep_separator=self.config.keep_separator,
                is_separator_regex=self.config.is_separator_regex,
                strip_whitespace=self.config.strip_whitespace
            ),
            'character': CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function,
                separator="\n\n"
            ),
            'token': TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        }
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = 'recursive',
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Chunk documents using specified strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy ('recursive', 'character', 'token', 'smart')
            preserve_metadata: Whether to preserve original metadata
            
        Returns:
            List of chunked documents
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
        
        try:
            logger.info(f"Chunking {len(documents)} documents with '{strategy}' strategy")
            
            if strategy == 'smart':
                chunks = self._smart_chunk_documents(documents, preserve_metadata)
            elif strategy in self.splitters:
                chunks = self._chunk_with_splitter(documents, strategy, preserve_metadata)
            else:
                logger.warning(f"Unknown strategy '{strategy}', using 'recursive'")
                chunks = self._chunk_with_splitter(documents, 'recursive', preserve_metadata)
            
            # Update statistics
            self._update_stats(documents, chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise RuntimeError(f"Document chunking failed: {str(e)}")
    
    def _chunk_with_splitter(
        self,
        documents: List[Document],
        strategy: str,
        preserve_metadata: bool
    ) -> List[Document]:
        """Chunk documents using specified text splitter."""
        splitter = self.splitters[strategy]
        chunks = []
        
        for i, doc in enumerate(documents):
            try:
                # Split document
                doc_chunks = splitter.split_documents([doc])
                
                # Add chunk metadata
                for j, chunk in enumerate(doc_chunks):
                    if preserve_metadata:
                        # Preserve original metadata and add chunk info
                        chunk.metadata.update({
                            'chunk_id': f"doc_{i}_chunk_{j}",
                            'chunk_index': j,
                            'total_chunks': len(doc_chunks),
                            'original_doc_index': i,
                            'chunking_strategy': strategy
                        })
                    
                    chunks.append(chunk)
                    
            except Exception as e:
                logger.error(f"Failed to chunk document {i}: {str(e)}")
                continue
        
        return chunks
    
    def _smart_chunk_documents(
        self,
        documents: List[Document],
        preserve_metadata: bool
    ) -> List[Document]:
        """
        Smart chunking that adapts to document structure.
        """
        chunks = []
        
        for i, doc in enumerate(documents):
            try:
                # Detect document type and structure
                doc_type = self._detect_document_type(doc)
                
                if doc_type == 'structured':
                    # Use section-based chunking
                    doc_chunks = self._chunk_by_sections(doc)
                elif doc_type == 'academic':
                    # Use paragraph-based chunking
                    doc_chunks = self._chunk_by_paragraphs(doc)
                else:
                    # Fall back to recursive chunking
                    doc_chunks = self.splitters['recursive'].split_documents([doc])
                
                # Add metadata
                for j, chunk in enumerate(doc_chunks):
                    if preserve_metadata:
                        chunk.metadata.update({
                            'chunk_id': f"doc_{i}_chunk_{j}",
                            'chunk_index': j,
                            'total_chunks': len(doc_chunks),
                            'original_doc_index': i,
                            'chunking_strategy': 'smart',
                            'detected_type': doc_type
                        })
                    
                    chunks.append(chunk)
                    
            except Exception as e:
                logger.error(f"Smart chunking failed for document {i}: {str(e)}")
                # Fall back to recursive chunking
                fallback_chunks = self.splitters['recursive'].split_documents([doc])
                chunks.extend(fallback_chunks)
        
        return chunks
    
    def _detect_document_type(self, doc: Document) -> str:
        """Detect document type based on content patterns."""
        content = doc.page_content
        
        # Check for academic paper structure
        academic_patterns = [
            r'\babstract\b',
            r'\bintroduction\b',
            r'\bmethodology\b',
            r'\bresults\b',
            r'\bconclusion\b',
            r'\breferences\b'
        ]
        
        academic_score = sum(1 for pattern in academic_patterns 
                           if re.search(pattern, content, re.IGNORECASE))
        
        # Check for structured document
        structured_patterns = [
            r'^#+\s',  # Markdown headers
            r'^\d+\.\s',  # Numbered sections
            r'^[A-Z][A-Z\s]+:',  # CAPITALIZED SECTIONS
        ]
        
        structured_score = sum(1 for pattern in structured_patterns 
                             if re.search(pattern, content, re.MULTILINE))
        
        if academic_score >= 3:
            return 'academic'
        elif structured_score >= 2:
            return 'structured'
        else:
            return 'general'
    
    def _chunk_by_sections(self, doc: Document) -> List[Document]:
        """Chunk document by sections/headers."""
        content = doc.page_content
        sections = re.split(r'\n(?=#+\s|^\d+\.\s|\n[A-Z][A-Z\s]+:)', content)
        
        chunks = []
        for section in sections:
            if len(section.strip()) > 50:  # Minimum section size
                # Further split if section is too large
                if len(section) > self.config.chunk_size:
                    sub_chunks = self.splitters['recursive'].split_text(section)
                    for sub_chunk in sub_chunks:
                        chunks.append(Document(
                            page_content=sub_chunk,
                            metadata=doc.metadata.copy()
                        ))
                else:
                    chunks.append(Document(
                        page_content=section,
                        metadata=doc.metadata.copy()
                    ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, doc: Document) -> List[Document]:
        """Chunk document by paragraphs with intelligent merging."""
        content = doc.page_content
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata=doc.metadata.copy()
                    ))
                    current_chunk = paragraph
                else:
                    # Single paragraph is too large, split it
                    sub_chunks = self.splitters['recursive'].split_text(paragraph)
                    for sub_chunk in sub_chunks:
                        chunks.append(Document(
                            page_content=sub_chunk,
                            metadata=doc.metadata.copy()
                        ))
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining content
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk,
                metadata=doc.metadata.copy()
            ))
        
        return chunks
    
    def _update_stats(self, original_docs: List[Document], chunks: List[Document]):
        """Update chunking statistics."""
        self.stats['total_documents'] += len(original_docs)
        self.stats['total_chunks'] += len(chunks)
        
        if chunks:
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            self.stats['average_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes)
            self.stats['min_chunk_size'] = min(self.stats['min_chunk_size'], min(chunk_sizes))
            self.stats['max_chunk_size'] = max(self.stats['max_chunk_size'], max(chunk_sizes))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        stats = self.stats.copy()
        if stats['min_chunk_size'] == float('inf'):
            stats['min_chunk_size'] = 0
        return stats
    
    def optimize_chunk_size(self, documents: List[Document]) -> int:
        """
        Analyze documents and suggest optimal chunk size.
        
        Args:
            documents: Documents to analyze
            
        Returns:
            Suggested chunk size
        """
        if not documents:
            return self.config.chunk_size
        
        # Calculate average document length
        doc_lengths = [len(doc.page_content) for doc in documents]
        avg_length = sum(doc_lengths) / len(doc_lengths)
        
        # Suggest chunk size based on document characteristics
        if avg_length < 1000:
            suggested_size = 300
        elif avg_length < 5000:
            suggested_size = 800
        elif avg_length < 20000:
            suggested_size = 1200
        else:
            suggested_size = 1500
        
        logger.info(f"Suggested chunk size: {suggested_size} (avg doc length: {avg_length:.0f})")
        return suggested_size
    
    def chunk_single_text(self, text: str, strategy: str = 'recursive') -> List[str]:
        """
        Chunk a single text string.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy
            
        Returns:
            List of text chunks
        """
        if strategy not in self.splitters:
            strategy = 'recursive'
        
        splitter = self.splitters[strategy]
        return splitter.split_text(text)
    
    def validate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Validate chunked documents and return quality metrics.
        
        Args:
            chunks: Chunked documents to validate
            
        Returns:
            Dictionary with validation results
        """
        if not chunks:
            return {'valid': False, 'reason': 'No chunks provided'}
        
        issues = []
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        # Check for empty chunks
        empty_chunks = sum(1 for size in chunk_sizes if size == 0)
        if empty_chunks > 0:
            issues.append(f"{empty_chunks} empty chunks found")
        
        # Check for oversized chunks
        oversized = sum(1 for size in chunk_sizes if size > self.config.chunk_size * 1.5)
        if oversized > 0:
            issues.append(f"{oversized} chunks exceed expected size")
        
        # Check for very small chunks
        tiny_chunks = sum(1 for size in chunk_sizes if size < 50)
        if tiny_chunks > len(chunks) * 0.1:  # More than 10% tiny chunks
            issues.append(f"{tiny_chunks} very small chunks (< 50 chars)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }


# Factory function for easy initialization
def create_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = 'recursive'
) -> DocumentChunker:
    """
    Create a document chunker with specified parameters.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        strategy: Default chunking strategy
        
    Returns:
        Configured DocumentChunker instance
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return DocumentChunker(config)