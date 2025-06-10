from dotenv import load_dotenv
load_dotenv()

# Initial Setup
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
# Preprocessing Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Processing document
# Tool
from langchain_core.tools import tool  # For creating tool
# History Persistence
from langgraph.checkpoint.memory import MemorySaver
# Agent 
from langgraph.prebuilt import create_react_agent


##############################
# Setting up model and tools #
##############################

# API Key prompt
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai") # Chat Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
vector_store = InMemoryVectorStore(embeddings) 

document_titles = []

print("Page to process >", end=" ")
rag_source = input()
while rag_source != "-1":
    web_loader = WebBaseLoader(rag_source)
    docs = web_loader.load()
    document_titles.append(docs[0].metadata.get('title'))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    document_ids = vector_store.add_documents(documents=all_splits)
    print("Page to process >", end=" ")
    rag_source = input()

print(f"Doc Titles: {document_titles}")
##############################
# Setting up the Application #
##############################


# Defining Document Retrieval Tool

@tool(response_format="content_and_artifact")
def retrieve_info(query: str):
    """Retrieve additional information that user would want the responses to be based on"""
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}" for doc in retrieved_docs))
    return serialized, retrieved_docs


# Create Agent with Persistence
system_prompt = f"You are given these documents {document_titles}.\n Determine if they are relevant to the question and, as much as possible, utilise the retrieve_info tool to answer the user's question"

memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve_info], checkpointer=memory, prompt=system_prompt)
config = {"configurable": {"thread_id": "def234"}}

while True:
    print(">", end=" ")
    input_message = input()
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()
