# Medical AI Assistant

A Retrieval-Augmented Generation (RAG) based Medical AI Assistant that provides general health information using Gemini API. This chatbot is designed to answer medical queries while maintaining appropriate disclaimers and scope limitations.

## ⚠️ Important Medical Disclaimer

**This is not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider. This assistant provides general health information only and should not replace professional medical advice.**

## 🏥 Features

- **RAG-based Architecture**: Uses Retrieval-Augmented Generation for accurate medical information
- **Openrouter API (Gemini 1.5 Flash) Integration**: Powered by Google's Gemini API for response generation
- **Medical Scope Limitation**: Only answers medical-related questions
- **Safety Disclaimers**: Includes appropriate medical disclaimers
- **Vector Search**: Uses FAISS for efficient similarity search
- **Streamlit UI**: Clean, user-friendly web interface

## 📁 Project Structure

```
Medical_AI_Assistant/
├── data/                          # Data directory
│   ├── processed_chunks.json     # Processed text chunks
│   ├── faiss_index.bin          # FAISS vector index
│   ├── chunks_data.json         # Chunk metadata
│   ├── embeddings.npy           # Vector embeddings
│   ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│   ├── model_info.json          # Model information
│   └── *.txt                    # Medical documents
├── src/                          # Source code
│   ├── ingestion.py             # Document cleaning and chunking
│   ├── embed_index.py           # Embeddings and FAISS index
│   ├── retriever.py             # Vector search logic
│   └── qa.py                    # Prompt composition and LLM call
├── examples/
│   └── sample_responses.md      # Example query outputs
├── Medical_AI_Assistant.py      # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- openrouter(Gemini API key)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashasvi1110/Medical_AI_Assistant.git
   cd Medical_AI_Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - The API key is already configured in the code

4. **Run the application**
   ```bash
   streamlit run Medical_AI_Assistant.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start asking medical questions!

## 🔧 Setup Instructions

### 1. Data Preparation

The project includes sample medical documents in the `data/` folder:
- `vitamin_b12_guide.txt`
- `fever_prevention_guide.txt`
- `first_aid_burns.txt`
- `dehydration_symptoms.txt`
- `stress_management.txt`
- `nutrition_basics.txt`

### 2. Processing Pipeline

1. **Document Ingestion** (`src/ingestion.py`)
   - Cleans and preprocesses medical documents
   - Splits text into 400-600 token chunks with overlap
   - Saves processed chunks with metadata

2. **Embedding Generation** (`src/embed_index.py`)
   - Creates TF-IDF embeddings for text chunks
   - Builds FAISS index for efficient similarity search
   - Saves embeddings and index for reuse

3. **Retrieval System** (`src/retriever.py`)
   - Implements vector search using FAISS
   - Returns top-k most relevant chunks
   - Handles query preprocessing

4. **Question Answering** (`src/qa.py`)
   - Composes prompts with retrieved context
   - Integrates with Gemini API
   - Includes safety disclaimers and scope limitations

### 3. Running the System

```bash
# Process documents (if needed)
python src/ingestion.py

# Create embeddings and index
python src/embed_index.py

# Run the main application
streamlit run Medical_AI_Assistant.py
```

## 🎯 Usage Examples

### Medical Questions
- "How to get rid of headache?"
- "What is vitamin B12 deficiency?"
- "Home remedies for fever"
- "Symptoms of dehydration"
- "How to manage stress?"

### Non-Medical Questions
The assistant will respond with: "This is not my expertise. I can only help with medical and health-related questions."

## 📊 Technical Details

### Architecture
- **RAG Pipeline**: Document → Chunks → Embeddings → FAISS Index → Retrieval → LLM
- **Embedding Model**: TF-IDF vectorization
- **Vector Database**: FAISS IndexFlatIP for cosine similarity
- **LLM**: Google Gemini 1.5 Flash
- **UI Framework**: Streamlit

### Key Components
- **Document Processing**: Text cleaning, chunking, metadata extraction
- **Vector Search**: FAISS-based similarity search
- **Response Generation**: Gemini API integration with context
- **Safety Measures**: Medical disclaimers, scope limitations

## 🛡️ Safety Features

1. **Medical Scope Only**: Declines non-medical questions
2. **Clear Disclaimers**: Prominent medical disclaimers
3. **Professional Advice**: Always recommends consulting healthcare providers
4. **Context-Aware**: Uses retrieved medical information for responses

## 📝 API Configuration

The system uses Google Gemini API with the following configuration:
- **Model**: `gemini-1.5-flash`
- **API Key**: Configured in the code
- **Rate Limiting**: Built-in API rate limiting
- **Error Handling**: Graceful fallback for API issues

## 🔍 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **API Key Issues**
   - Verify Gemini API key is valid
   - Check API quota and billing

3. **FAISS Index Issues**
   - Recreate index: `python src/embed_index.py`
   - Check data directory permissions

### Performance Tips

- Use SSD storage for FAISS index
- Monitor API usage and costs
- Adjust chunk size for your use case

## 📈 Future Enhancements

- [ ] Support for more document formats
- [ ] Advanced embedding models
- [ ] Multi-language support
- [ ] Conversation history
- [ ] Export functionality
- [ ] Mobile-responsive design

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical regulations in your jurisdiction.

## ⚠️ Important Notes

- **Not a Medical Device**: This is not a medical device or diagnostic tool
- **Educational Purpose**: Intended for educational and informational purposes only
- **Professional Consultation**: Always consult qualified healthcare providers for medical advice
- **Data Privacy**: Be mindful of sensitive medical information

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the example responses in `examples/sample_responses.md`

---


**Remember**: This assistant provides general health information only. For any medical concerns, please consult a qualified healthcare provider.
