"""
Medical AI Assistant - Streamlit Cloud Entry Point
Handles missing dependencies gracefully
"""
import streamlit as st
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    st.warning("⚠️ Some features may be limited due to missing dependencies.")

try:
    import faiss
    import sentence_transformers
    HAS_ML = True
except ImportError:
    HAS_ML = False
    st.warning("⚠️ RAG features disabled - using simplified mode.")

# Import and run the main app
if __name__ == "__main__":
    # Import the main function from Medical_AI_Assistant.py
    exec(open('Medical_AI_Assistant.py').read())
