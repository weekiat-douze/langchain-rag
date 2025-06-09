from dotenv import load_dotenv
load_dotenv()

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
# Creating Prompt Template
from langchain_core.prompts import PromptTemplate
# Tools
from langchain_core.tools import tool  # For creating tool
# Agent
from langgraph.prebuilt import create_react_agent
# For LangChain persistence — checkpointer
from langgraph.checkpoint.memory import MemorySaver

##############################
# Setting up model and tools #
##############################

# API Key prompt
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai") # Chat Model

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Allow us to generate embeddings for Gemini
# To generate embeddings of supplementary documents to be used in RAG 

vector_store = InMemoryVectorStore(embeddings) # Utilise in-memory vector store for supplementary document embeddings
# Here we we indicate to use GoogleGenerativeAIEmbedding in the vector store

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

# assert len(docs) == 1
# print(f"Total characters: {len(docs[0].page_content)}")


"""
Processing the document
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# print(f"Split blog post into {len(all_splits)} sub-documents.")

"""
Embedding & Storage
"""
document_ids = vector_store.add_documents(documents=all_splits)

# print(document_ids[:3])

#############
# Retrieval #
#############
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search( query, k=2 )
    serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}" for doc in retrieved_docs))
    return serialized, retrieved_docs

memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)


#########################
# Using the Application #
#########################
config = {"configurable": {"thread_id": "def234"}}

# input_message = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )

input_message = "What is Task Decomposition?"

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()

input_message = "Can you look up some common ways of doing it?"

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()
