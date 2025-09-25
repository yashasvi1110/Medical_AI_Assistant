"""
Fixed RAG-based Medical AI Assistant using Gemini API
Uses urllib instead of requests to avoid conflicts.
"""

import streamlit as st
import os
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Any


class FixedRAGGeminiMedicalChatbot:
    """Fixed RAG-based medical chatbot using Gemini API with urllib."""
    
    def __init__(self):
        """Initialize the fixed RAG Gemini medical chatbot."""
        self.disclaimer = "⚠️ **IMPORTANT DISCLAIMER**: I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
        
        # Your Gemini API key
        self.api_key = "AIzaSyAaWucUUbxdeRPjkcwSi6TRFNAIv1U7mmw"
        self.model_name = "gemini-1.5-flash"
        
        # Medical knowledge base for RAG
        self.medical_knowledge = {
            "headache": {
                "info": "Headache relief: rest in dark room, cold compress, hydration, gentle neck stretches, OTC pain relievers. See doctor for severe or frequent headaches.",
                "sources": "headache_guide"
            },
            "fever": {
                "info": "Fever treatment: rest, hydration, cool compresses, fever reducers, monitor temperature. Seek medical attention if fever is high or persistent.",
                "sources": "fever_guide"
            },
            "stomach ache": {
                "info": "Stomach relief: bland foods, clear liquids, avoid spicy foods, peppermint tea, gentle heat. Seek medical attention if pain is severe.",
                "sources": "digestive_guide"
            },
            "period pain": {
                "info": "Period pain relief: heat therapy, gentle exercise, OTC pain relievers, relaxation techniques, avoid caffeine/alcohol. Consult doctor if severe.",
                "sources": "women_health_guide"
            },
            "knee pain": {
                "info": "Knee pain relief: rest, ice/heat therapy, gentle stretching, supportive footwear, avoid high-impact activities. Consult doctor for persistent pain.",
                "sources": "joint_health_guide"
            },
            "back pain": {
                "info": "Back pain relief: rest, ice/heat therapy, gentle stretching, good posture, firm mattress, avoid heavy lifting. Consult doctor for persistent pain.",
                "sources": "back_health_guide"
            },
            "cold": {
                "info": "Cold treatment: rest, hydration, saline nasal spray, humidifier, honey (adults), 7-10 days recovery. See doctor if symptoms persist.",
                "sources": "respiratory_health_guide"
            },
            "cough": {
                "info": "Cough relief: warm liquids, humidifier, honey (adults), salt water gargle, throat lozenges, avoid irritants. See doctor if cough persists.",
                "sources": "respiratory_health_guide"
            },
            "sore throat": {
                "info": "Sore throat relief: salt water gargle, warm liquids, throat lozenges, rest voice, avoid irritants. See doctor if symptoms persist.",
                "sources": "throat_health_guide"
            },
            "nausea": {
                "info": "Nausea relief: small bland meals, clear liquids, avoid strong smells, ginger tea, comfortable position. Seek medical attention if severe.",
                "sources": "digestive_health_guide"
            },
            "fatigue": {
                "info": "Fatigue management: adequate sleep, regular schedule, hydration, balanced meals, exercise, stress management. Consult doctor if persistent.",
                "sources": "energy_health_guide"
            },
            "stress": {
                "info": "Stress management: deep breathing, meditation, exercise, sleep, limit caffeine, talk to others, professional counseling if needed.",
                "sources": "mental_health_guide"
            },
            "insomnia": {
                "info": "Sleep improvement: regular schedule, comfortable environment, avoid screens/caffeine, relaxation techniques. Consult doctor for persistent problems.",
                "sources": "sleep_health_guide"
            },
            "anxiety": {
                "info": "Anxiety management: deep breathing, meditation, exercise, sleep, limit caffeine, talk to others, professional counseling if needed.",
                "sources": "mental_health_guide"
            },
            "vitamin b12": {
                "info": "Vitamin B12: essential for nerve function, found in animal products, deficiency causes fatigue/weakness. Consult doctor for testing and supplementation.",
                "sources": "nutrition_guide"
            },
            "dehydration": {
                "info": "Dehydration prevention: drink water, eat water-rich foods, avoid excess caffeine/alcohol, watch for thirst/dark urine. Seek medical attention if severe.",
                "sources": "hydration_guide"
            }
        }
        
        # Medical keywords
        self.medical_keywords = [
            'health', 'medical', 'disease', 'symptom', 'treatment', 'medicine',
            'vitamin', 'nutrition', 'exercise', 'prevention', 'first aid',
            'fever', 'pain', 'injury', 'burn', 'dehydration', 'stress',
            'headache', 'cough', 'cold', 'flu', 'diabetes', 'blood pressure',
            'heart', 'lung', 'stomach', 'skin', 'bone', 'muscle', 'joint',
            'allergy', 'infection', 'wound', 'cut', 'bruise', 'swelling',
            'nausea', 'dizziness', 'fatigue', 'sleep', 'diet', 'weight',
            'home remedy', 'natural treatment', 'herbal', 'supplement',
            'ache', 'sore', 'throat', 'back', 'neck', 'shoulder', 'knee',
            'ankle', 'wrist', 'elbow', 'hip', 'chest', 'abdomen',
            'tired', 'weak', 'dizzy', 'nauseous', 'vomiting', 'diarrhea',
            'constipation', 'bloating', 'gas', 'indigestion', 'heartburn',
            'insomnia', 'anxiety', 'depression', 'mood', 'mental health',
            'breathing', 'shortness', 'wheezing', 'asthma', 'allergy',
            'rash', 'itchy', 'redness', 'swelling', 'inflammation',
            'cramps', 'spasms', 'stiffness', 'tension', 'soreness',
            'period', 'menstrual', 'cramps', 'pms'
        ]
        
        # Non-medical keywords
        self.non_medical_keywords = [
            'mathematics', 'math', 'physics', 'chemistry', 'biology',
            'history', 'geography', 'politics', 'government', 'election',
            'weather', 'climate', 'sports', 'football', 'cricket', 'basketball',
            'entertainment', 'movie', 'music', 'dance', 'art', 'painting',
            'cooking', 'recipe', 'food recipe', 'restaurant', 'cuisine',
            'travel', 'tourism', 'vacation', 'hotel', 'booking',
            'shopping', 'buy', 'sell', 'price', 'cost', 'money',
            'technology', 'computer', 'programming', 'coding', 'software',
            'finance', 'banking', 'investment', 'stock', 'trading',
            'education', 'school', 'college', 'university', 'study',
            'job', 'career', 'employment', 'work', 'business'
        ]
    
    def is_medical_query(self, query: str) -> bool:
        """Check if query is medical-related."""
        query_lower = query.lower()
        
        # Check for medical keywords
        has_medical_keywords = any(keyword in query_lower for keyword in self.medical_keywords)
        
        # Check for non-medical keywords
        has_non_medical_keywords = any(keyword in query_lower for keyword in self.non_medical_keywords)
        
        # If it has non-medical keywords and no medical keywords, it's not medical
        if has_non_medical_keywords and not has_medical_keywords:
            return False
        
        # If it has medical keywords, it's medical
        if has_medical_keywords:
            return True
        
        # Check for common medical question patterns
        medical_patterns = [
            'what is', 'how to', 'symptoms of', 'treatment for', 'cure for',
            'prevent', 'avoid', 'home remedy', 'natural treatment',
            'should i', 'can i', 'is it safe', 'side effects',
            'causes of', 'signs of', 'warning signs', 'when to see'
        ]
        
        has_medical_patterns = any(pattern in query_lower for pattern in medical_patterns)
        
        return has_medical_patterns
    
    def retrieve_medical_context(self, query: str) -> str:
        """Retrieve relevant medical context for RAG."""
        query_lower = query.lower()
        
        # Check for specific conditions
        for condition, info in self.medical_knowledge.items():
            if condition in query_lower:
                return info['info']
        
        # Check for related terms
        if any(term in query_lower for term in ['head', 'head pain', 'migraine']):
            return self.medical_knowledge['headache']['info']
        elif any(term in query_lower for term in ['stomach', 'belly', 'tummy', 'abdominal']):
            return self.medical_knowledge['stomach ache']['info']
        elif any(term in query_lower for term in ['knee', 'knees']):
            return self.medical_knowledge['knee pain']['info']
        elif any(term in query_lower for term in ['back', 'spine']):
            return self.medical_knowledge['back pain']['info']
        elif any(term in query_lower for term in ['period', 'menstrual', 'cramps']):
            return self.medical_knowledge['period pain']['info']
        elif any(term in query_lower for term in ['fever', 'temperature', 'hot']):
            return self.medical_knowledge['fever']['info']
        elif any(term in query_lower for term in ['cough', 'coughing']):
            return self.medical_knowledge['cough']['info']
        elif any(term in query_lower for term in ['throat', 'sore throat']):
            return self.medical_knowledge['sore throat']['info']
        elif any(term in query_lower for term in ['nausea', 'nauseous']):
            return self.medical_knowledge['nausea']['info']
        elif any(term in query_lower for term in ['tired', 'fatigue', 'exhausted']):
            return self.medical_knowledge['fatigue']['info']
        elif any(term in query_lower for term in ['stress', 'anxiety', 'worried']):
            return self.medical_knowledge['stress']['info']
        elif any(term in query_lower for term in ['sleep', 'insomnia', 'sleepless']):
            return self.medical_knowledge['insomnia']['info']
        elif any(term in query_lower for term in ['b12', 'b-12', 'vitamin b12']):
            return self.medical_knowledge['vitamin b12']['info']
        elif any(term in query_lower for term in ['water', 'thirst', 'dehydration']):
            return self.medical_knowledge['dehydration']['info']
        
        return "General medical information"
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using urllib."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            # Convert data to JSON
            json_data = json.dumps(data).encode('utf-8')
            
            # Create request
            req = urllib.request.Request(
                f"{url}?key={self.api_key}",
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )
            
            # Make request
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['candidates'][0]['content']['parts'][0]['text']
                
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function using RAG with Gemini API."""
        try:
            # Check if query is medical
            if not self.is_medical_query(query):
                return {
                    'response': "I'm a medical information assistant and can only help with health-related questions. Please ask me about medical topics, symptoms, treatments, or general health information.",
                    'is_medical': False,
                    'sources': [],
                    'disclaimer': self.disclaimer,
                    'model': self.model_name,
                    'rag': True
                }
            
            # Retrieve medical context (RAG)
            medical_context = self.retrieve_medical_context(query)
            
            # Create prompt for Gemini
            prompt = f"""You are a helpful medical information assistant. Provide general health information and home remedies based on the medical context below.

MEDICAL CONTEXT: {medical_context}

USER QUESTION: {query}

IMPORTANT RULES:
1. Only provide general health information and home remedies
2. Always include this disclaimer: "⚠️ IMPORTANT DISCLAIMER: I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
3. Do not provide specific medical diagnoses or prescriptions
4. Encourage consulting healthcare professionals for serious conditions
5. Focus on general wellness and home care tips
6. Keep response concise but helpful

Please provide a helpful response with general health information and home remedies."""

            # Call Gemini API
            response_text = self.call_gemini_api(prompt)
            
            # Ensure disclaimer is included
            if self.disclaimer not in response_text:
                response_text = f"{response_text}\n\n{self.disclaimer}"
            
            return {
                'response': response_text,
                'is_medical': True,
                'sources': ['rag_medical_knowledge'],
                'disclaimer': self.disclaimer,
                'model': self.model_name,
                'rag': True
            }
            
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error while processing your request. Please try again or consult a healthcare professional.\n\n{self.disclaimer}",
                'is_medical': True,
                'sources': [],
                'disclaimer': self.disclaimer,
                'model': self.model_name,
                'rag': True,
                'error': str(e)
            }


def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FixedRAGGeminiMedicalChatbot()
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


def display_disclaimer():
    """Display the medical disclaimer banner."""
    st.markdown("""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336; margin-bottom: 20px;">
        <h4 style="color: #d32f2f; margin: 0;">⚠️ Important Medical Disclaimer</h4>
        <p style="margin: 10px 0 0 0; color: #424242;">
            <strong>I am not a medical professional.</strong> For diagnosis or treatment, consult a qualified healthcare provider. 
            This assistant provides general health information only and should not replace professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_conversation():
    """Display the conversation history."""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q: {exchange['query'][:50]}...", expanded=False):
                st.markdown(f"**Question:** {exchange['query']}")
                st.markdown(f"**Answer:** {exchange['response']}")
                if 'sources' in exchange and exchange['sources']:
                    st.markdown(f"**Sources:** {', '.join(exchange['sources'])}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏥 Medical AI Assistant")
    st.markdown("**Powered by Gemini 1.5 Flash with RAG** - Ask me about medical topics, symptoms, treatments, or general health information")
    
    # Display disclaimer
    display_disclaimer()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        st.success("✅ Gemini API Ready")
        st.markdown(f"**Model:** {st.session_state.chatbot.model_name}")
        st.markdown("**API:** Google Gemini")
        st.markdown("**RAG:** Retrieval-Augmented Generation")
        st.markdown("**Medical Knowledge:** Enhanced")
        st.markdown("**HTTP Library:** urllib (No conflicts)")
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💬 Recent Questions")
            for exchange in st.session_state.conversation_history[-5:]:
                st.markdown(f"• {exchange['query'][:30]}...")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Medical Question")
        
        # Query input
        query = st.text_area(
            "Enter your health-related question:",
            placeholder="e.g., How to get rid of headache? What is vitamin B12? Home remedies for fever?",
            height=100,
            help="Ask about medical topics, symptoms, treatments, or general health information."
        )
        
        # Submit button
        if st.button("🔍 Get Medical Answer", type="primary"):
            if not query.strip():
                st.warning("Please enter a medical question.")
            else:
                with st.spinner("Generating response with Gemini AI and RAG..."):
                    # Get response from chatbot
                    response_data = st.session_state.chatbot.chat(query)
                    
                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'query': query,
                        'response': response_data['response'],
                        'sources': response_data.get('sources', []),
                        'is_medical': response_data.get('is_medical', False),
                        'model': response_data.get('model', 'unknown'),
                        'rag': response_data.get('rag', False)
                    })
                    
                    st.rerun()
    
    with col2:
        st.header("📊 Response Info")
        
        if st.session_state.conversation_history:
            latest = st.session_state.conversation_history[-1]
            
            # Display latest response
            st.markdown("### Latest Response")
            st.markdown(latest['response'])
            
            # Display metadata
            if 'sources' in latest and latest['sources']:
                st.markdown(f"**Sources:** {len(set(latest['sources']))} documents")
                st.markdown(f"**RAG:** {'Yes' if latest.get('rag') else 'No'}")
                st.markdown(f"**AI Model:** {latest.get('model', 'Unknown')}")
            
            # Show if it's medical or not
            if latest.get('is_medical'):
                st.success("✅ Medical Query Processed")
            else:
                st.warning("⚠️ Non-Medical Query Declined")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        display_conversation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>Medical AI Assistant | Powered by Gemini 1.5 Flash with RAG | For general health information only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
