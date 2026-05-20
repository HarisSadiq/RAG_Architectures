from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


app = FastAPI()


@app.get("/")
def message():

    return {
        "message": "API is running"
    }


# LOAD EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LOAD CHROMADB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# LOAD LLM
llm = ChatOllama(
    model="phi3",
    temperature=0
)


# REQUEST FORMAT
class Query(BaseModel):

    question: str = Field(..., example="summary of research paper",
        description="User question for the RAG chatbot")

    


# CHAT ENDPOINT
@app.post("/chat")
def chat(user_query: Query):


    q = user_query.question


    docs = vectorstore.similarity_search(
        q,
        k=3
    )


    context = "\n\n".join([
        doc.page_content for doc in docs
    ])


    final_prompt = f"""
    Answer the user's question ONLY from the context below.

    Context:
    {context}

    Question:
    {q}
    """


    response = llm.invoke(final_prompt)


    return {
        "question": q,
        "answer": response.content
    }