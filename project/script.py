from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_doc():
    path = "data_set.pdf"

    loader = PyPDFLoader(path)

    data = loader.load()

    print(f"Total pages loaded: {len(data)}")

    return data

documents = load_doc()


def chunk_data(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    
    return chunks
chunks = chunk_data(documents)

def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nEmbedding model loaded successfully")

    return embeddings

embeddings=load_embedding_model()


# STEP 4 — Convert Chunks into Embeddings
def store_in_chroma(chunks, embeddings):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("\nChunks stored successfully in ChromaDB")

    return vectorstore




vectorstore = store_in_chroma(chunks, embeddings)



