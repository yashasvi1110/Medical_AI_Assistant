"""
Vector retrieval module for Medical AI Assistant.
Handles semantic search and retrieval of relevant document chunks.
"""

import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
import os


class MedicalRetriever:
    """Handles semantic search and retrieval of medical information."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the medical retriever.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        self.index = None
        self.chunks_data = []
        self.dimension = None
        self.is_loaded = False
    
    def load_index(self, index_path: str = "data/faiss_index.bin", 
                   chunks_path: str = "data/chunks_data.json") -> bool:
        """
        Load FAISS index and chunks data.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load chunks data
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks_data = json.load(f)
            
            # Load model info
            with open('data/model_info.json', 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.dimension = model_info['dimension']
            
            self.is_loaded = True
            print(f"Successfully loaded index with {self.index.ntotal} vectors")
            print(f"Loaded {len(self.chunks_data)} chunks")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.is_loaded:
            raise ValueError("Index not loaded. Please call load_index() first.")
        
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve chunks and filter by score
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= min_score and idx < len(self.chunks_data):
                chunk = self.chunks_data[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Chunk dictionary if found, None otherwise
        """
        for chunk in self.chunks_data:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific source document.
        
        Args:
            source: Source document identifier
            
        Returns:
            List of chunks from the specified source
        """
        return [chunk for chunk in self.chunks_data if chunk['source'] == source]
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available source documents.
        
        Returns:
            List of source document names
        """
        sources = set(chunk['source'] for chunk in self.chunks_data)
        return sorted(list(sources))
    
    def search_with_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search with additional context information.
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            Dictionary containing results and metadata
        """
        results = self.search(query, k)
        
        # Add source information
        sources = {}
        for result in results:
            source = result['source']
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            'query': query,
            'results': results,
            'total_results': len(results),
            'sources_covered': list(sources.keys()),
            'source_counts': sources
        }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate if a query is appropriate for medical information retrieval.
        
        Args:
            query: User query to validate
            
        Returns:
            Dictionary with validation results
        """
        # Basic validation rules
        medical_keywords = [
            'health', 'medical', 'disease', 'symptom', 'treatment', 'medicine',
            'vitamin', 'nutrition', 'exercise', 'prevention', 'first aid',
            'fever', 'pain', 'injury', 'burn', 'dehydration', 'stress'
        ]
        
        query_lower = query.lower()
        
        # Check for medical relevance
        has_medical_keywords = any(keyword in query_lower for keyword in medical_keywords)
        
        # Check for inappropriate requests
        inappropriate_patterns = [
            'diagnose', 'prescription', 'medicine for', 'what medicine',
            'doctor', 'clinic', 'hospital', 'emergency'
        ]
        
        has_inappropriate = any(pattern in query_lower for pattern in inappropriate_patterns)
        
        return {
            'is_valid': has_medical_keywords and not has_inappropriate,
            'has_medical_keywords': has_medical_keywords,
            'has_inappropriate': has_inappropriate,
            'suggested_action': 'proceed' if has_medical_keywords and not has_inappropriate else 'refuse'
        }


def main():
    """Main function to test the retriever."""
    retriever = MedicalRetriever()
    
    # Load index
    if not retriever.load_index():
        print("Failed to load index. Please run embed_index.py first.")
        return
    
    # Test queries
    test_queries = [
        "How to prevent viral fever",
        "What is vitamin B12",
        "Foods high in B12",
        "Symptoms of dehydration",
        "Basic first-aid steps for a minor burn"
    ]
    
    print("\nTesting retrieval system:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Validate query
        validation = retriever.validate_query(query)
        print(f"Valid: {validation['is_valid']}")
        
        if validation['is_valid']:
            # Search for relevant chunks
            results = retriever.search(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['similarity_score']:.3f}")
                print(f"   Source: {result['source']}")
                print(f"   Text: {result['text'][:100]}...")
                print()
        else:
            print("Query not appropriate for medical information retrieval.")


if __name__ == "__main__":
    main()
