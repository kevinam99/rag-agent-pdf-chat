import os
from typing import Annotated, Sequence, TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from .system_prompt import SYSTEM_MESSAGE

models = {"qwen3": "qwen3:1.7b", "nomic": "nomic-embed-text:v1.5"}

