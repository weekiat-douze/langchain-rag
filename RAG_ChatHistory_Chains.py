from dotenv import load_dotenv
load_dotenv() # Needs to be at the top because LangChain checks for USER_AGENT on package import

# Initial Setup
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Web Page Retrieval
import bs4   # Web Scraping package
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Processing document
# Retrieval, Augmentation & Generation
from langchain import hub # This is to utilise LangChain Hub
from langchain_core.documents import Document # Typing for state definition
from typing_extensions import List, TypedDict # Typing for state definition
from langgraph.graph import START, StateGraph # LangGraph
# For Typing in Search class
from typing import Literal
from typing_extensions import Annotated
# Chains Historical Message
from langgraph.graph import MessagesState # 
from langchain_core.tools import tool  # For creating tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
# For LangChain persistence — checkpointer
from langgraph.checkpoint.memory import MemorySaver

##############################
# Setting up model and tools #
##############################

# API Key prompt
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Setup Model, Embedding Function, Vector Store
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai") # Chat Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
vector_store = InMemoryVectorStore(embeddings) 

#########################
# Actual RAG Operations # 
#########################

###################################################
# Pre-processing Step for Supplementary Documents #
###################################################

"""
Retrieval of Web Page 
"""

# WebBaseLoader makes use of BS4 within to parse the webpage 
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content")) # Sets up filtering here, Only keep post title, headers, and content from the full HTML.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()


"""
Processing the document
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)


"""
Embedding & Storage
"""
document_ids = vector_store.add_documents(documents=all_splits)


#############
# Retrieval #
#############


# Setup Nodes for LangGraph
"""
Nodes — Return dictionary and LangGraph will handle reconcilliation with the State
"""
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search( query, k=2 )
    serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}" for doc in retrieved_docs))
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools = llm.bind_tools([retrieve])    # This binding will expose the tool to the llm, which decides if it wants to invoke
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

    
tools = ToolNode([retrieve]) # This Node will execute the retrieval (the tool)



def generate(state: MessagesState):
    """Generate answer."""
    # Get ToolMessages — in this case is the RAG retrieval
    recent_tool_messages = []
    for message in reversed(state['messages']):
        if message.type == "tool":
            recent_tool_messages.append(message) # Get the latest tool messages (multiple if consecutive)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    
    # RAG Prompt — updated for Historical Messages
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ('human', 'system')
        or (message,type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    
    # Invoking LLM
    response = llm.invoke(prompt)
    return { "messages": [response]}


# Set up graph flow for the application

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond") # this eqvuialent to add_edge(START, <node>)
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver() # for persistence
graph = graph_builder.compile(checkpointer=memory) 

# Specify an ID for the thread — persistence
config = {"configurable": {"thread_id": "abc123"}}



#########################
# Using the Application #
#########################
# input_message = "Hello"

# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
    
    
input_message = "What is Task Decomposition?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()
    
input_message = "Can you look up some common ways of doing it?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()
