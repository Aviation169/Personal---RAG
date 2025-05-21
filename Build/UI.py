import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import gc
import os
from io import StringIO
import torch

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 10px;
    }
    .stButton > button {
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #1e40af;
    }
    .answer-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .source-text {
        font-style: italic;
        color: #4b5563;
        margin-bottom: 10px;
    }
    .sidebar .stFileUploader {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .history-card {
        background-color: #f9fafb;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border-left: 4px solid #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning("No GPU detected. Running on CPU.")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None

# Initialize models and vector store based on uploaded file
@st.cache_resource
def initialize_models(uploaded_file_content, _cache_key):
    """
    Initialize models and vector store using the uploaded file content.
    _cache_key is used to differentiate cache based on file content.
    """
    temp_file_path = "temp_uploaded_news.txt"
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(uploaded_file_content)

    loader = TextLoader(temp_file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Initialize embeddings with GPU support
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vector_db = FAISS.from_documents(chunks, embedding_model)

    # Initialize LLM with GPU support
    llm = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",  # Replace with your valid model
        device=0 if device == "cuda" else -1,  # 0 for GPU, -1 for CPU
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
    )

    os.remove(temp_file_path)
    return vector_db, embedding_model, llm

def process_query(query, vector_db, llm, conversation_history):
    """Processes the query with conversation history and returns the response."""
    retrieved_docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    history_text = ""
    if conversation_history:
        history_text = "\n".join([f"Q: {item['query']}\nA: {item['response']}" for item in conversation_history])
        history_text += "\n\n"

    prompt = f"Previous conversation:\n{history_text}Context from document:\n{context}\n\nCurrent Question: {query}\nAnswer:"
    response = llm(prompt, max_new_tokens=300)[0]["generated_text"]

    answer_start = response.find("Answer:") + len("Answer:")
    response = response[answer_start:].strip()

    gc.collect()
    return response

# Streamlit UI
st.title("Personal RAG with Session Memory (GPU-Enabled)")

# Sidebar for file upload and instructions
with st.sidebar:
    st.header("Upload News File")
    st.write("Upload a .txt file to query its content.")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"], key="file_uploader")
    st.markdown("---")
    st.write("**Instructions**:")
    st.write("- Upload a UTF-8 encoded text file.")
    st.write("- Enter a question about the file's content.")
    st.write("- Click 'Get Answer' to see the response.")
    st.write("- Use 'Reset' to clear the query, answer, and conversation history.")

# Main content
if uploaded_file is not None:
    try:
        file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        file_name = uploaded_file.name
        file_hash = hash(file_content)

        if st.session_state.current_file_hash != file_hash:
            st.session_state.conversation_history = []
            st.session_state.current_file_hash = file_hash

        cache_key = file_hash
        vector_db, embedding_model, llm = initialize_models(file_content, cache_key)
        with st.status("Processing file...", expanded=True) as status:
            st.write("Loading file content...")
            st.write("Creating vector store...")
            st.write("Initializing language model...")
            status.update(label="File processed successfully!", state="complete")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.stop()
else:
    st.info("Please upload a text file in the sidebar to proceed.")
    st.stop()

# Query input and buttons
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Your Question:", value="Is there any girl name mentioned here?", key="query_input")
with col2:
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    reset = st.button("Reset")

# Handle reset
if reset:
    st.session_state.query_input = ""
    st.session_state.response = None
    st.session_state.conversation_history = []
    st.rerun()

# Process query
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
                response = process_query(query, vector_db, llm, st.session_state.conversation_history)
                st.session_state.conversation_history.append({"query": query, "response": response})
                st.session_state.response = response
                st.session_state.file_name = file_name
            except Exception as e:
                st.error(f"Error processing query: {e}")
    else:
        st.warning("Please enter a valid question.")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### Conversation History")
    for i, item in enumerate(st.session_state.conversation_history):
        st.markdown(
            f"<div class='history-card'>"
            f"<strong>Question {i+1}:</strong> {item['query']}<br>"
            f"<strong>Answer:</strong> {item['response']}"
            f"</div>",
            unsafe_allow_html=True
        )

# Display latest response
if "response" in st.session_state and st.session_state.response:
    st.markdown(f"<div class='source-text'>Answer based on: {st.session_state.file_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-card'><strong>Latest Answer:</strong><br>{st.session_state.response}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Built with Streamlit and LangChain. © 2025")
