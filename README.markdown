# Llama3 RAG System

The Llama3 RAG System is a Retrieval-Augmented Generation (RAG) application that allows users to upload text documents, index them, and query information using a combination of document retrieval and language model generation. It leverages LangChain, FAISS, and Hugging Face models to provide accurate, context-aware responses. The system offers two user interfaces:

1. **HTML/CSS/JavaScript UI**: A responsive web interface served by Flask, featuring a sidebar, chat area, and modals for document uploads and settings.
2. **Streamlit UI**: A simple, interactive interface built with Streamlit, ideal for quick prototyping and testing.

Both UIs interact with a Flask backend API, running on CPU with `faiss-cpu` for stability.

## Features
- **Document Upload**: Upload `.txt` files to build a knowledge base.
- **RAG Pipeline**: Retrieves relevant document chunks using FAISS and generates answers with a lightweight LLM (`llama3.2`).
- **Conversation History**: Tracks queries and responses per session.
- **Session Management**: Reset sessions to clear history and vector stores.
- **Dual Interfaces**:
  - **HTML UI**: Rich, customizable interface with Tailwind CSS, modals, and dynamic chat updates.
  - **Streamlit UI**: Minimalist interface for uploading documents and querying, with a focus on ease of use.

## Prerequisites
- Python 3.8+
- Git
- A Hugging Face account and API token for model access (e.g., `meta-llama/Llama-3.2-3B-Instruct`, `all-MiniLM-L6-v2`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Aviation169/Personal---RAG.git
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `env.example` to `.env` and add your Hugging Face token:
   ```bash
   cp ../env.example .env
   ```
   Edit `.env`:
   ```
   HF_TOKEN=your_huggingface_token_here
   FLASK_ENV=development
   FLASK_DEBUG=1
   ```

## Running the Application

### Option 1: HTML/CSS/JavaScript UI (Flask)
The Flask server hosts both the API and the HTML-based UI at `http://localhost:5000`.

1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
   Output:
   ```
   Running on device: cpu
    * Serving Flask app 'app'
    * Debug mode: on
    * Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
   ```

2. **Access the UI**:
   Open a browser and navigate to `http://localhost:5000`. You’ll see a web interface with:
   - A sidebar for starting new chats, viewing recent chats, and uploading documents.
   - A chat area with a welcome message and conversation history.
   - Modals for uploading `.txt` files and adjusting settings (placeholder).

### Option 2: Streamlit UI
The Streamlit app provides a simpler interface at `http://localhost:8501`, communicating with the Flask API.

1. **Start the Flask API**:
   The Streamlit UI requires the Flask API to be running. In one terminal:
   ```bash
   python app.py
   ```

2. **Start the Streamlit App**:
   In another terminal:
   ```bash
   streamlit run UI.py
   ```
   Output:
   ```
     You can now view your Streamlit app in your browser.
     Local URL: http://localhost:8501
   ```

3. **Access the UI**:
   Open `http://localhost:8501`. The interface includes:
   - A file uploader for `.txt` files.
   - A text input for queries.
   - A display area for responses and conversation history.

## Usage

### HTML UI
1. **Upload a Document**:
   - Click “Upload Documents” in the sidebar.
   - Select a `.txt` file and click “Upload Documents.”
   - Wait for the notification: “Documents uploaded successfully!”

2. **Ask Questions**:
   - Type a query in the input area (e.g., “Is there any girl name mentioned here?”).
   - Click the send button or press Enter.
   - View the response in the chat area, with the source document linked.

3. **Manage History**:
   - Click “Recent Chats” to reload past conversations.
   - Click “Clear” to reset the session.

4. **Settings**:
   - Click “Settings” to open the modal (currently a placeholder for model and RAG configurations).

### Streamlit UI
1. **Upload a Document**:
   - Use the file uploader to select a `.txt` file.
   - Click “Upload” to process the file.
   - See a success message or error.

2. **Ask Questions**:
   - Enter a query in the text input.
   - Click “Submit” to get a response.
   - View the response and source document below.

3. **View History**:
   - Past queries and responses are displayed in the app.
   - Click “Reset Session” to clear the session.

## RAG Overview
Retrieval-Augmented Generation (RAG) combines document retrieval with language model generation:
- **Retrieval**: Uses FAISS to index document chunks (via `all-MiniLM-L6-v2` embeddings) and retrieve relevant chunks based on query similarity.
- **Generation**: Feeds retrieved chunks and query to an LLM (`meta-llama/Llama-3.2-3B-Instruct`) to generate context-aware answers.
- **Benefits**: Enhances answer accuracy by grounding responses in uploaded documents, reducing hallucination.

## Troubleshooting
- **“This site can’t be reached” (ERR_CONNECTION_REFUSED)**:
  - Ensure Flask (`python app.py`) or Streamlit (`streamlit run UI.py`) is running.
  - Check port 5000 (Flask) or 8501 (Streamlit):
    ```bash
    netstat -tuln | grep 5000
    netstat -tuln | grep 8501
    ```
  - Resolve port conflicts:
    ```bash
    sudo lsof -i :5000
    sudo kill -9 <PID>
    ```
  - Allow ports:
    ```bash
    sudo ufw allow 5000
    sudo ufw allow 8501
    ```

- **Model Loading Errors**:
  - Verify `HF_TOKEN` in `.env`.
  - Use a lighter model in `app.py`:
    ```python
    llm = pipeline("text-generation", model="facebook/opt-125m", device=-1) 
    ```

- **Upload Errors**:
  - Ensure `.txt` files are UTF-8 encoded:
    ```bash
    file -i your_file.txt
    ```
  - Check Flask logs for errors.

- **Streamlit API Errors**:
  - Ensure Flask API is running at `http://localhost:5000` before starting Streamlit.
  - Verify `API_URL = "http://localhost:5000"` in `UI.py`.

## Future Enhancements
- Add SQLite for persistent history.
- Implement document modal content in HTML UI.
- Enable settings functionality (e.g., temperature, retrieval count).
- Support GPU execution with `faiss-gpu`.

## Contact
For issues or contributions, open a GitHub issue or contact [akajay14955j@gmail.com].