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
# prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com") # Get the prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = PromptTemplate.from_template(template)

# Defining State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Setup Nodes for LangGraph
"""
Nodes — Return dictionary and LangGraph will handle reconcilliation with the State
"""

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # invoke() doesn't call the LLM, it populates the placeholders for the prompt template
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages) # Actual LLM Call
    return {"answer": response.content}


# Set up graph flow for the application
graph_builder = StateGraph(State).add_sequence([retrieve, generate]) 
    # `StateGraph()` <- initial setup 
    # `add_sequence` <- add the nodes, and connects them in a linear manner (adding edge)
graph_builder.add_edge(START, "retrieve") # Adding edge from START to retrieve (START is special entry point)
graph = graph_builder.compile() # Actually building the flow

#########################
# Using the Application #
#########################
result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
