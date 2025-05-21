from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch
import faiss
import gc
import os
from io import StringIO
import uuid
from typing import List, Dict

app = Flask(__name__, template_folder="templates", static_folder="static")

# Force CPU usage
device = "cpu"

# In-memory storage for vector stores and conversation history
vector_stores = {}  # session_id -> FAISS vector store
conversation_history = {}  # session_id -> List[Dict[str, str]]
embedding_model = None

# Initialize embedding model
def initialize_embedding_model():
    global embedding_model
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    print(f"Running on device: {device}")

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

# Serve index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    file = request.files['file']
    if not file.filename.endswith(".txt"):
        return jsonify({"detail": "Only .txt files are supported"}), 400
    
    try:
        session_id = str(uuid.uuid4())
        content = StringIO(file.read().decode("utf-8")).read()
        
        # Save to temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{session_id}.txt")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Load and split document
        loader = TextLoader(temp_file_path, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create FAISS index on CPU
        vector_db = FAISS.from_documents(chunks, embedding_model)
        
        vector_stores[session_id] = vector_db
        conversation_history[session_id] = []
        
        os.remove(temp_file_path)
        return jsonify({"session_id": session_id, "filename": secure_filename(file.filename)})
    except Exception as e:
        return jsonify({"detail": f"Error processing file: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    session_id = data.get('session_id')
    query = data.get('query', '').strip()
    
    if session_id not in vector_stores:
        return jsonify({"detail": "Session not found"}), 404
    if not query:
        return jsonify({"detail": "Query cannot be empty"}), 400
    
    try:
        vector_db = vector_stores[session_id]
        # Initialize LLM on CPU
        llm = pipeline(
            "text-generation",
            model="G:\My Drive\llama-3.2-3b-instruct",  # Replace with your valid model
            device=-1,  # -1 for CPU
        )
        
        # Retrieve context
        retrieved_docs = vector_db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Build prompt with conversation history
        history_text = ""
        if session_id in conversation_history and conversation_history[session_id]:
            history_text = "\n".join([f"Q: {item['query']}\nA: {item['response']}" 
                                     for item in conversation_history[session_id]])
            history_text += "\n\n"
        
        prompt = f"Previous conversation:\n{history_text}Context from document:\n{context}\n\nCurrent Question: {query}\nAnswer:"
        response = llm(prompt, max_new_tokens=300)[0]["generated_text"]
        
        # Extract answer
        answer_start = response.find("Answer:") + len("Answer:")
        response_text = response[answer_start:].strip()
        
        # Update conversation history
        conversation_history[session_id].append({"query": query, "response": response_text})
        
        # Clean up
        gc.collect()
        
        return jsonify({"response": response_text, "source": "Uploaded document"})
    except Exception as e:
        return jsonify({"detail": f"Error processing query: {str(e)}"}), 500

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    if session_id not in conversation_history:
        return jsonify({"detail": "Session not found"}), 404
    return jsonify({"history": conversation_history[session_id]})

@app.route('/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    if session_id not in vector_stores:
        return jsonify({"detail": "Session not found"}), 404
    vector_stores.pop(session_id, None)
    conversation_history.pop(session_id, None)
    gc.collect()
    return jsonify({"message": "Session reset successfully"})

@app.route('/device-status', methods=['GET'])
def device_status():
    return jsonify({
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

if __name__ == '__main__':
    initialize_embedding_model()
    app.run(host='0.0.0.0', port=5000, debug=True)