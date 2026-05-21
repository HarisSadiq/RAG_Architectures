from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

import numpy as np



# =====================================================
# STEP 1 — LOAD EMBEDDING MODEL
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding Model Loaded Successfully")


# =====================================================
# STEP 2 — LOAD CHROMADB
# =====================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("\nChromaDB Loaded Successfully")


# =====================================================
# STEP 3 — LOAD ALL CHUNKS FROM CHROMADB
# =====================================================

all_data = vectorstore.get()

documents = all_data["documents"]

print(f"\nTotal Chunks Loaded: {len(documents)}")


# =====================================================
# STEP 4 — CREATE BM25 INDEX
# =====================================================

tokenized_docs = [
    doc.split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)

print("\nBM25 Retriever Created Successfully")


# =====================================================
# STEP 5 — LOAD RERANKER MODEL
# =====================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("\nReranker Loaded Successfully")


# =====================================================
# STEP 6 — LOAD OLLAMA LLM
# =====================================================

llm = ChatOllama(
    model="phi3",
    temperature=0
)

print("\nLLM Loaded Successfully")


# =====================================================
# STEP 7 — START CHATBOT LOOP
# =====================================================

print("\nHybrid RAG Chatbot Ready")


while True:

    query = input("\nAsk Your Question: ")


    # EXIT CONDITION
    if query.lower() == "exit":

        print("\nGoodbye!")

        break


    # =====================================================
    # STEP 8 — SEMANTIC SEARCH
    # =====================================================

    semantic_results = vectorstore.similarity_search(
        query,
        k=5
    )

    semantic_chunks = [
        doc.page_content
        for doc in semantic_results
    ]

    print("\nSemantic Search Completed")


    # =====================================================
    # STEP 9 — BM25 KEYWORD SEARCH
    # =====================================================

    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    top_bm25_indices = np.argsort(
        bm25_scores
    )[-5:][::-1]

    bm25_chunks = [
        documents[i]
        for i in top_bm25_indices
    ]

    print("BM25 Search Completed")


    # =====================================================
    # STEP 10 — MERGE RESULTS
    # =====================================================

    combined_chunks = semantic_chunks + bm25_chunks


    # REMOVE DUPLICATES
    combined_chunks = list(
        set(combined_chunks)
    )

    print(f"\nTotal Retrieved Chunks: {len(combined_chunks)}")


    # =====================================================
    # STEP 11 — RERANKING
    # =====================================================

    pairs = [
        [query, chunk]
        for chunk in combined_chunks
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    reranked_results = sorted(
        zip(combined_chunks, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nReranking Completed")


    # =====================================================
    # STEP 12 — SELECT TOP CHUNKS
    # =====================================================

    top_chunks = reranked_results[:3]

    final_context = "\n\n".join([
        chunk[0]
        for chunk in top_chunks
    ])


    # =====================================================
    # STEP 13 — CREATE FINAL PROMPT
    # =====================================================

    final_prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question ONLY
    from the provided context.

    If the answer is not present
    in the context, say:

    "I could not find the answer in the document."


    Context:
    {final_context}


    Question:
    {query}
    """


    # =====================================================
    # STEP 14 — GENERATE RESPONSE
    # =====================================================

    response = llm.invoke(
        final_prompt
    )


    # =====================================================
    # STEP 15 — PRINT FINAL ANSWER
    # =====================================================

    print("\n================ ANSWER ================\n")

    print(response.content)