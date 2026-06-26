import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

def get_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_answer(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the PDF context below.
If the answer is not in the context, say "I don't know based on the PDF."

Context:
{context}

Question: {question}
Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    response = model.invoke([HumanMessage(content=prompt)])
    if isinstance(response.content, list):
        return response.content[0].get("text", str(response.content))
    return response.content

# --- UI ---
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

with st.sidebar:
    st.header("Upload PDF")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Process PDF"):
        if pdf:
            with st.spinner("Reading and indexing PDF..."):
                raw_text = get_pdf_text(pdf)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDF processed! Ask your questions.")
        else:
            st.warning("Please upload a PDF first.")

question = st.text_input("Ask a question from your PDF:")
if question:
    with st.spinner("Thinking..."):
        answer = get_answer(question)
        st.write("**Answer:**", answer)