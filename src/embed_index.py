"""
Embedding generation and FAISS indexing module for Medical AI Assistant.
Handles vector embeddings creation and storage for semantic search.
"""

import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")


class EmbeddingIndexer:
    """Handles embedding generation and FAISS index creation using TF-IDF approach."""
    
    def __init__(self, model_name: str = "tfidf"):
        """
        Initialize the embedding indexer.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.embeddings = None
        self.index = None
        self.chunks_data = []
        
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            numpy array of embeddings
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        texts = [chunk['text'] for chunk in chunks]
        
        # Create TF-IDF embeddings
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray().astype('float32')
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            FAISS index
        """
        print("Creating FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)
        
        print(f"FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, chunks_data: List[Dict], 
                   embeddings: np.ndarray, output_dir: str = "data"):
        """
        Save FAISS index and related data.
        
        Args:
            index: FAISS index
            chunks_data: List of chunk data
            embeddings: numpy array of embeddings
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
        
        # Save chunks data
        with open(os.path.join(output_dir, "chunks_data.json"), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        
        # Save vectorizer
        with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "embedding_dimension": embeddings.shape[1],
            "num_chunks": len(chunks_data),
            "index_type": "FAISS_IndexFlatIP"
        }
        
        with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Index and data saved to {output_dir}/")
    
    def load_index(self, data_dir: str = "data"):
        """
        Load FAISS index and related data.
        
        Args:
            data_dir: Directory containing saved files
            
        Returns:
            tuple of (index, chunks_data, embeddings, vectorizer)
        """
        # Load FAISS index
        index = faiss.read_index(os.path.join(data_dir, "faiss_index.bin"))
        
        # Load chunks data
        with open(os.path.join(data_dir, "chunks_data.json"), 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Load embeddings
        embeddings = np.load(os.path.join(data_dir, "embeddings.npy"))
        
        # Load vectorizer
        with open(os.path.join(data_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        
        self.vectorizer = vectorizer
        self.embeddings = embeddings
        self.index = index
        self.chunks_data = chunks_data
        
        print(f"Loaded index with {index.ntotal} vectors and {len(chunks_data)} chunks")
        return index, chunks_data, embeddings, vectorizer
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the loaded index.
        
        Args:
            query: Query text
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunks with metadata
        """
        if self.index is None or self.vectorizer is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Transform query to embedding
        query_embedding = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Get similar chunks
        similar_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_data):
                chunk = self.chunks_data[idx].copy()
                chunk['similarity_score'] = float(score)
                similar_chunks.append(chunk)
        
        return similar_chunks


def main():
    """Main function to create and save embeddings."""
    # Load processed chunks
    with open("data/processed_chunks.json", 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Create embeddings
    indexer = EmbeddingIndexer()
    embeddings = indexer.create_embeddings(chunks)
    
    # Create FAISS index
    index = indexer.create_faiss_index(embeddings)
    
    # Save everything
    indexer.save_index(index, chunks, embeddings)
    
    print("Embedding index created and saved successfully!")


if __name__ == "__main__":
    main()