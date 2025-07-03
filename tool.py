from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

@tool
def retriever_tool(retriever: VectorStoreRetriever, query: str) -> str:
    """This tools searches and returns the information from the given document"""

    docs = retriever.invoke(query)

    if not docs:
        return "Found no relevant information in the given file"
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    
    return "\n\n".join(results)