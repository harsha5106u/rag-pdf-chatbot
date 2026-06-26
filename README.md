# RAG PDF Chatbot

An AI-powered chatbot that answers questions from any PDF document using Retrieval-Augmented Generation (RAG).

## Tech Stack
- Python
- LangChain
- Google Gemini API
- FAISS (Vector Store)
- Streamlit

## How it works
1. Upload any PDF document
2. Document is chunked and converted to embeddings
3. Embeddings stored in FAISS vector store
4. User asks a question
5. Relevant chunks retrieved and passed to Gemini LLM
6. Answer generated based on document context

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google API key in `.env` file as `GOOGLE_API_KEY=your_key`
4. Run: `streamlit run chatbotai.py`