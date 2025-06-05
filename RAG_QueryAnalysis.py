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
# Creating Prompt Template
from langchain_core.prompts import PromptTemplate
# For Typing in Search class
from typing import Literal
from typing_extensions import Annotated

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

#### Adding additional metadata so we can utilise them for Query Analysis
third = len(all_splits) // 3 # split document into thirds
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

"""
Embedding & Storage
"""
document_ids = vector_store.add_documents(documents=all_splits)


#############
# Retrieval #
#############
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = PromptTemplate.from_template(template)

# Defining a Structure Output schema
class Search(TypedDict):
    """Search query.""" 
    query: Annotated[str, ..., "Search query to run."] # These are type definition syntax
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Defining State
class State(TypedDict):
    question: str
    query: Search         # Updating container to include data for Query Analysis
    context: List[Document]
    answer: str

# Setup Nodes for LangGraph
"""
Nodes — Return dictionary and LangGraph will handle reconcilliation with the State
"""
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search) # Create a separate LLM that will utilise the Schema for Structured Output
    query = structured_llm.invoke(state["question"]) # Calling LLM
    return { "query" : query } # Update state with result


def retrieve(state: State):
    query = state["query"] # retrieve the result from analyze_query operation
    retrieved_docs = vector_store.similarity_search(
        state["question"],
        filter=lambda doc: doc.metadata.get("section") == query["section"] # only keep documents that matches with analyze_query
        )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages) # Actual LLM Call
    return {"answer": response.content}


# Set up graph flow for the application
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate]) 
graph_builder.add_edge(START, "analyze_query") 
graph = graph_builder.compile() 



#########################
# Using the Application #
#########################
result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Section: {result["query"]}\n\n')
print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
