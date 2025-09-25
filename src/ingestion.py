"""
Document ingestion and preprocessing module for Medical AI Assistant.
Handles document cleaning, chunking, and metadata extraction.
"""

import os
import re
import json
import tiktoken
from typing import List, Dict, Any
from pathlib import Path


class DocumentProcessor:
    """Handles document cleaning and chunking operations."""
    
    def __init__(self, chunk_size: int = 500, overlap_size: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks in tokens
            overlap_size: Overlap between consecutive chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common document artifacts
        text = re.sub(r'^[A-Z\s]{10,}$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Cleaned text content
            source: Source document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self.count_tokens(test_chunk)
            
            if token_count > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_id': f"{source}_{chunk_id}",
                    'source': source,
                    'text': current_chunk.strip(),
                    'token_count': self.count_tokens(current_chunk),
                    'chunk_index': chunk_id
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = test_chunk
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{source}_{chunk_id}",
                'source': source,
                'text': current_chunk.strip(),
                'token_count': self.count_tokens(current_chunk),
                'chunk_index': chunk_id
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from the end of a chunk."""
        words = text.split()
        overlap_words = words[-self.overlap_size:] if len(words) > self.overlap_size else []
        return " ".join(overlap_words)
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Clean the text
        cleaned_content = self.clean_text(content)
        
        # Extract source name from file path
        source = Path(file_path).stem
        
        # Create chunks
        chunks = self.create_chunks(cleaned_content, source)
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all text files in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all processed chunks from all documents
        """
        all_chunks = []
        
        for file_path in Path(directory_path).glob("*.txt"):
            print(f"Processing: {file_path}")
            chunks = self.process_document(str(file_path))
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        Save processed chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the chunks
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to process documents."""
    processor = DocumentProcessor(chunk_size=500, overlap_size=100)
    
    # Process documents from data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        chunks = processor.process_directory(data_dir)
        
        # Save processed chunks
        output_path = "data/processed_chunks.json"
        processor.save_chunks(chunks, output_path)
        
        print(f"\nProcessing complete!")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(c['token_count'] for c in chunks) / len(chunks):.1f} tokens")
    else:
        print(f"Data directory '{data_dir}' not found. Please add some .txt files to process.")


if __name__ == "__main__":
    main()
