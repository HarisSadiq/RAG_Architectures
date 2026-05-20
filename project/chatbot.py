from dotenv import load_dotenv

from langchain_ollama import ChatOllama

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import HumanMessage


# STEP 1 — Load API Key
load_dotenv()


# STEP 2 — Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 3 — Load Existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("\nChromaDB Loaded Successfully")


# STEP 4 — Load OpenAI Model



# STEP 3 — Load Local LLM
llm = ChatOllama(
    model="phi3",
    temperature=0
)

print("Local LLM Loaded Successfully")

print("\nRAG Chatbot Ready")


# STEP 5 — Chat Loop
while True:

    query = input("\nAsk Your Question: ")


    # Exit condition
    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # STEP 6 — Retrieve Relevant Chunks
    docs = vectorstore.similarity_search(query, k=3)


    # STEP 7 — Combine Retrieved Context
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])


    # STEP 8 — Create Prompt
    final_prompt = f"""
    Answer the user's question ONLY from the context below.

    Context:
    {context}

    Question:
    {query}
    """


    # STEP 9 — Generate Answer
    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])


    # STEP 10 — Print Answer
    print("\nANSWER:\n")

    print(response.content)