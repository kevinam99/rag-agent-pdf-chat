from typing import Annotated, Sequence, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

from system_prompt import SYSTEM_MESSAGE
from utils import MODELS, load_pdf, initialise_vector_store, vectorise_document


pages = load_pdf()
vector_store = initialise_vector_store()
vectorise_document(vector_store, pages)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """This tools searches and returns the information from the given document"""

    docs = retriever.invoke(query)

    if not docs:
        return "Found no relevant information in the given file"
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]
# temperature = 0 to reduce hallucination
llm = ChatOllama(model=MODELS["qwen3"], temperature=0).bind_tools(tools) 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> AgentState:
    """Checks if the last message has tool calls. If yes, continue, else, end"""
    result = state["messages"][-1]

    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


def call_llm(state: AgentState) -> AgentState:
    """Function to call LLM with current state"""

    messages = [SYSTEM_MESSAGE] + state["messages"]
    message = llm.invoke(messages)

    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute the tool calls from the LLMs responses"""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    tools_dict = {our_tool.name: our_tool for our_tool in tools}

    for t in tool_calls:
        print(f"Calling tool {t['name']} with query {t['args'].get('query', 'No query found')}")

        if not t['name'] in tools_dict:
            print(f"{t['name']} does not exist")
            result = "Incorrect tool name. Please retry and select the tool available from the list of tools"
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ""))
            print(f"Result length: {len(str(result))}")
        
        tool_message = ToolMessage(content=str(result), tool_call_id=t['id'], name=t['name'])
        results.append(tool_message)
    
    print("Tool execution complete, return to LLM")
    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)

graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
graph.add_edge("retriever", "llm")

graph.set_entry_point("llm")

rag_agent = graph.compile()


def run_agent():
    print("\n=== RAG AGENT===")

    while True:
        human_input = input("\nEnter your question: ")
        if human_input.lower() in ['exit', 'quit']:
            break
        
        human_message = HumanMessage(content=human_input)

        result = rag_agent.invoke({"messages": [human_message]})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)
