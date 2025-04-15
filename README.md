🤖Personal RAG prototype🤖
-

**⚠️For production, it required a lot of enhancement⚠️**

1️⃣A Streamlit-based web application implementing **Retrieval-Augmented Generation (RAG)** to answer questions about uploaded news content. 
The app retrieves relevant text from a user-provided .txt file using FAISS and LangChain, then generates answers with a language model.

📃Overview📃
-

This app leverages the RAG framework:

🫴Retrieval: Uses FAISS vector store and Hugging Face embeddings `(all-MiniLM-L6-v2)` to find relevant text chunks from an uploaded file.

📂Augmentation: Combines retrieved context with the user’s question to form a prompt.

📝Generation: Generates answers using a language model `(default: Llama3.2 for testing)`.

🖨️Features🖨️
-

€ Upload a `.txt file` to query its news content.

€ Ask questions and get answers powered by RAG.

€ Clean UI with a sidebar for file uploads, styled answer display, and a reset option.

€ Displays the source file name (e.g., "Answer based on: Latest_news.txt") instead of raw context.

⏬Prerequisites⏬
-

`Python 3.8+`

`Git installed`

`A GitHub account`

(Optional) A local or Hugging Face language model for generation

🧪Usage🧪
-

1️⃣Run the Streamlit app:

`streamlit run app.py`

2️⃣Open your browser to `http://localhost:8501`.

3️⃣In the sidebar, upload a UTF-8 encoded .txt file containing news content.

4️⃣Enter a question (e.g., "Is there any girl name mentioned here?").

5️⃣Click "Get Answer" to see the RAG-generated response, with the file name as the source.

6️⃣Click "Reset" to clear the query and answer.

💀RAG Configuration💀
-

→Embedding Model: `all-MiniLM-L6-v2` for efficient text embeddings.

→Vector Store: FAISS with `chunk_size=500` and `chunk_overlap=100` for retrieval.

→Language Model: Default is `Llama3.2`. To use a custom model, update `UI.py and Rag.ipynb`:

```
llm = pipeline("text-generation", model="your/model/path")
```

→Example for Hugging Face (requires authentication):

```
llm = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", token="your_hf_token")
```

→File Encoding: Uploaded files must be UTF-8 encoded to avoid errors.

(●'◡'●)License
-

MIT License
