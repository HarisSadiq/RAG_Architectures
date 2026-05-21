from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



def load_doc():
    path="data.pdf"
    loader=PyPDFLoader(path)
    data=loader.load()
    return data

docs=load_doc()

def chunks_data(docs):
    splitter=RecursiveCharacterTextSplitter(

        chunk_size=500,
        chunk_overlap=50
    )

    chunk=splitter.split_documents(docs)

    return chunk

chunks=chunks_data(docs)

def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nEmbedding model loaded successfully")

    return embeddings
embeddings=load_embedding_model()


def save_data(chunks,embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("\nChunks stored successfully in ChromaDB")

    return vectorstore
data=save_data(chunks,embeddings)

