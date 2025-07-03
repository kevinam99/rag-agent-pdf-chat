import os
from typing import Iterator
from langchain_ollama import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

MODELS = {"qwen3": "qwen3:1.7b", "nomic": "nomic-embed-text:v1.5"}

def load_pdf():
    pdf_path = "./Stock_Market_Performance_2024.pdf"

    # checks if path exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pdf_loader = PyPDFLoader(pdf_path)

    # lazy load to avoid loading entire file to memory
    try:
        pages = pdf_loader.lazy_load()
        print(f"PDF loaded lazily")
        return pages
    except Exception as e:
        print(f"Error loading pdf; {str(e)}")
        raise


def initialise_vector_store():
    # the embedding model must be compatible with the LLM
    embeddings = OllamaEmbeddings(model=MODELS["nomic"], temperature=0)
    return Chroma(embedding_function=embeddings, persist_directory='.', collection_name='stock_market')


def vectorise_document(vector_store: Chroma, pages: Iterator[Document]):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for page in pages:
            chunks = text_splitter.split_documents([page])
            vector_store.add_documents(chunks)
        print(f"Vector store created successfully")
    except Exception as e:
        print(f"Error setting up vector store: {str(e)}")
        raise
