

## Learning LangChain and Retrieval Augmented Generation (RAG)
-  Reference: [Part 1](https://python.langchain.com/docs/tutorials/rag/), [Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/)

I have used a new file for each section which I found was significant, and also included 2 files which are modifications to play around with RAG.

Experimenting Files:
- `RAG_Ollama.py` — RAG with Ollama so that the program is fully local
- `RAG_Gemini.py` — Modified tutorial to accept URL inputs for RAG


*LangChain tutorial on RAG also exposes us to techniques and features of LangChain, I'm including a high-level overview of it for reference

## Part 1 — RAG

- Building RAG (`RAG.py`)
    - Basics of implementing RAG with LangChain — Document Embedding, Setup LangGraph nodes
- Query Analysis (`RAG_QueryAnalysis.py`)
    - Using LLM + Structured Output to refine Retrieval during RAG

## Part 2 — Historical Messages
Incorporating historical messages in the app, with RAG context

- Chains (`RAG_ChatHistory_Chains.py`)
    - Using `MessagesState` and Checkpointer to retain history
    - Additionally introduced tool calling in LangChain
- Agents (`RAG_ChatHistory_Agent.py`)
    - Introduces LangGraph's pre-built agent constructor
    - Underlying implementation of agent allows for checkpointer to retain history 
