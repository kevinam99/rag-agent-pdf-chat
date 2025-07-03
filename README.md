# 💬 Chat with Your PDF — A Custom RAG App

Interact with your PDF files locally using a Retrieval-Augmented Generation (RAG) pipeline powered by [Ollama](https://ollama.com) and [LangGraph](https://github.com/langchain-ai/langgraph).

This project was created as a hands-on way to deepen my understanding of LangGraph and re-familiarize myself with Python development.

## 🔍 What This Project Covers

This app explores key concepts behind building LLM-powered RAG systems:

- 📄 **Chunking** — breaking PDFs into manageable pieces  
- 🧠 **Embeddings** — converting text into vector representations  
- 🗃️ **Vectorization** — indexing and retrieving relevant chunks for responses  

Overall, this was a fun and insightful way to dive into building local LLM applications using modern tools.

## 🚀 Getting Started

> **Note:** Make sure you have Python 3.13+ installed and [Ollama](https://ollama.com/) running with your preferred model.

1. **Clone the repository**
2. **Setup a virtual environment**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3. **Configure your settings**  
    Edit [config.py](./config.py) to adjust PDF file settings as needed. Defaults are provided for quick starts.
    The [utils.py](./utils.py) file has the models being used. You can add your models as you need.


4. **Run the app**
    ```python
    python main.py
    ```

## 🛠 Tech Stack
- Python 3.13
- Ollama — to run LLMs locally
- LangGraph — for building structured LLM workflows
- Chroma — utilities for embeddings, vector stores
- LngChain - document processing

📁 Project Goals

This is a personal learning project aimed at understanding the internals of RAG systems and LangGraph workflows. PRs, feedback or suggestions are welcome!