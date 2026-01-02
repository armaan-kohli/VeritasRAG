import os
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory containing your source files (PDFs, txt, md, etc.)
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def ingest_docs():
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not set. Cannot generate embeddings.")
        return
    
    # 1. Load Documents
    print(f"--- LOADING DOCUMENTS FROM {DATA_PATH} ---")
    if not os.path.exists(DATA_PATH):
        print(f"Directory {DATA_PATH} does not exist. Please create it and add files.")
        return

    # Automatically detects .txt and .pdf files
    # For more complex loading, you can use specific loaders per file type
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    raw_docs = []
    try:
        raw_docs.extend(loader.load())
        print(f"Loaded {len(raw_docs)} text docs")
    except Exception as e:
        print(f"Could not load text files: {e}")
        
    try:
        pdf_docs = pdf_loader.load()
        raw_docs.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF docs")
    except Exception as e:
        print(f"Could not load PDF files: {e}")

    if not raw_docs:
        print("No documents found. Add .txt or .pdf files to the 'data' folder.")
        return

    # 2. Split Documents (Critical for RAG)
    # Breaking large docs into smaller 'chunks' ensures the context fits in the LLM window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"--- SPLIT INTO {len(chunks)} CHUNKS ---")

    # 3. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. Create/Update Vector Store
    # Check if DB exists to avoid duplicate stacking if running multiple times (optional logic)
    if os.path.exists(DB_PATH):
        # Optional: Clear existing DB to start fresh
        # shutil.rmtree(DB_PATH) 
        pass

    print("--- INDEXING CHUNKS (This may take a moment) ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"--- FINISHED INGESTING {len(chunks)} CHUNKS ---")

if __name__ == "__main__":
    ingest_docs()
