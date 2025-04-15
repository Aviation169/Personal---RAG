import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import gc
import os
from io import StringIO

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
    </style>
""", unsafe_allow_html=True)

# Initialize models and vector store based on uploaded file
@st.cache_resource
def initialize_models(uploaded_file_content, _cache_key):
    """
    Initialize models and vector store using the uploaded file content.
    _cache_key is used to differentiate cache based on file content.
    """
    # Save uploaded file content to a temporary file
    temp_file_path = "temp_uploaded_news.txt"
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(uploaded_file_content)

    # Load and split document
    loader = TextLoader(temp_file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Create vector database
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embedding_model)

    # Initialize LLM (replace with your valid model path or Hugging Face model name)
    llm = pipeline("text-generation", model="G:\My Drive\llama-3.2-3b-instruct")  # Using distilgpt2 for testing
    # Example: llm = pipeline("text-generation", model="G:\My Drive\llama-3.2-3b-instruct")

    # Clean up temporary file
    os.remove(temp_file_path)

    return vector_db, embedding_model, llm

def process_query(query, vector_db, llm):
    """Processes the query and returns the response."""
    # Similarity search
    retrieved_docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response using LLM
    prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt, max_new_tokens=300)[0]["generated_text"]

    # Clean up memory
    gc.collect()

    return response

# Streamlit UI
st.title("Personal ~ RAG")

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
    st.write("- Use 'Reset' to clear the query and answer.")

# Main content
if uploaded_file is not None:
    try:
        # Read the uploaded file
        file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        file_name = uploaded_file.name
        # Use file content as a cache key to reinitialize models if file changes
        cache_key = hash(file_content)
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
    st.rerun()

# Process query
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
                response = process_query(query, vector_db, llm)
                st.session_state.response = response
                st.session_state.file_name = file_name
            except Exception as e:
                st.error(f"Error processing query: {e}")
    else:
        st.warning("Please enter a valid question.")

# Display response
if "response" in st.session_state and st.session_state.response:
    st.markdown(f"<div class='source-text'>Answer based on: {st.session_state.file_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-card'><strong>Answer:</strong><br>{st.session_state.response}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Built with Streamlit and LangChain. © 2025")