"""
Question-Answering module for Medical AI Assistant.
Handles LLM integration with safety disclaimers and context-aware responses.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from retriever import MedicalRetriever


class MedicalQA:
    """Handles question-answering with safety constraints and context."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the Medical QA system.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.retriever = MedicalRetriever()
        self.conversation_history = []
        
        # Safety disclaimers and prompts
        self.disclaimer = "⚠️ **IMPORTANT DISCLAIMER**: I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
        
        self.system_prompt = """You are a helpful medical information assistant. Your role is to provide general health information based on the provided context. 

CRITICAL SAFETY RULES:
1. Always include the disclaimer: "I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
2. Only provide general health information, never specific medical advice
3. Do not provide diagnoses, prescriptions, or treatment recommendations
4. If asked about specific medical conditions, symptoms, or treatments, redirect to healthcare professionals
5. If the query is outside your knowledge base, say: "This query is outside my knowledge base. Please consult an appropriate source."

Use only the provided context to answer questions. If the context doesn't contain relevant information, say so clearly."""

    def load_retriever(self) -> bool:
        """Load the retrieval system."""
        return self.retriever.load_index()
    
    def is_medical_query(self, query: str) -> bool:
        """
        Check if a query is appropriate for medical information.
        
        Args:
            query: User query
            
        Returns:
            True if appropriate for medical information
        """
        validation = self.retriever.validate_query(query)
        return validation['is_valid']
    
    def is_out_of_scope(self, query: str) -> bool:
        """
        Check if a query is completely out of scope.
        
        Args:
            query: User query
            
        Returns:
            True if query is out of scope
        """
        out_of_scope_keywords = [
            'weather', 'politics', 'sports', 'entertainment', 'cooking recipe',
            'travel', 'shopping', 'technology', 'programming', 'finance'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in out_of_scope_keywords)
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(f"Source {i} ({source}, relevance: {score:.2f}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, max_chunks: int = 5) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Check if query is out of scope
        if self.is_out_of_scope(query):
            return {
                'response': "This query is outside my knowledge base. Please consult an appropriate source.",
                'sources': [],
                'disclaimer': self.disclaimer,
                'is_out_of_scope': True,
                'confidence': 0.0
            }
        
        # Check if query is medical
        if not self.is_medical_query(query):
            return {
                'response': "I can only provide general health information. Please ask about health-related topics or consult a healthcare professional for medical advice.",
                'sources': [],
                'disclaimer': self.disclaimer,
                'is_medical': False,
                'confidence': 0.0
            }
        
        # Retrieve relevant chunks
        chunks = self.retriever.search(query, k=max_chunks, min_score=0.1)
        
        if not chunks:
            return {
                'response': "I don't have specific information about this topic in my knowledge base. Please consult a healthcare professional for accurate medical information.",
                'sources': [],
                'disclaimer': self.disclaimer,
                'confidence': 0.0
            }
        
        # Format context
        context = self.format_context(chunks)
        
        # Create user prompt
        user_prompt = f"""Context Information:
{context}

User Question: {query}

Please provide a helpful response based on the context above. Remember to include the disclaimer and only provide general health information."""
        
        try:
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Add disclaimer if not already present
            if self.disclaimer not in answer:
                answer = f"{self.disclaimer}\n\n{answer}"
            
            return {
                'response': answer,
                'sources': [chunk['source'] for chunk in chunks],
                'chunks': chunks,
                'disclaimer': self.disclaimer,
                'confidence': max(chunk.get('similarity_score', 0) for chunk in chunks),
                'is_medical': True,
                'is_out_of_scope': False
            }
            
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error while processing your request. Please try again or consult a healthcare professional.",
                'sources': [],
                'disclaimer': self.disclaimer,
                'error': str(e),
                'confidence': 0.0
            }
    
    def add_to_conversation(self, query: str, response: str):
        """Add query and response to conversation history."""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': None  # Could add timestamp if needed
        })
        
        # Keep only last 10 exchanges to manage context length
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_context(self) -> str:
        """Get recent conversation context for multi-turn conversations."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"Q: {exchange['query']}")
            context_parts.append(f"A: {exchange['response']}")
        
        return "\n".join(context_parts)
    
    def generate_contextual_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response with conversation context.
        
        Args:
            query: Current user query
            
        Returns:
            Response dictionary with context
        """
        # Get conversation context
        context = self.get_conversation_context()
        
        # Generate response
        response_data = self.generate_response(query)
        
        # Add to conversation history
        self.add_to_conversation(query, response_data['response'])
        
        return response_data


def main():
    """Main function to test the QA system."""
    qa = MedicalQA()
    
    # Load retriever
    if not qa.load_retriever():
        print("Failed to load retriever. Please run embed_index.py first.")
        return
    
    # Test queries
    test_queries = [
        "How to prevent viral fever",
        "What is vitamin B12",
        "Foods high in B12",
        "Symptoms of dehydration",
        "Basic first-aid steps for a minor burn",
        "Is exercise good for managing stress",
        "Should I take antibiotics for a sore throat",  # Should be refused
        "What is the weather today"  # Should be refused as out-of-scope
    ]
    
    print("Testing Medical QA System:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        response_data = qa.generate_response(query)
        
        print(f"Response: {response_data['response']}")
        print(f"Sources: {response_data['sources']}")
        print(f"Confidence: {response_data['confidence']:.3f}")
        print(f"Medical: {response_data.get('is_medical', False)}")
        print(f"Out of scope: {response_data.get('is_out_of_scope', False)}")


if __name__ == "__main__":
    main()
