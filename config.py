MODELS = {
    "qwen3": "qwen3:1.7b", 
    "nomic": "nomic-embed-text:v1.5"
    }

PDF_FILE_PATH = "./Stock_Market_Performance_2024.pdf"
PERSIST_EMBEDDINGS_DIRECTORY = "."
EMBEDDINGS_COLLECTION_NAME = "stock_market"

LLM = MODELS["qwen3"]
EMBEDDING_MODEL = MODELS["nomic"]