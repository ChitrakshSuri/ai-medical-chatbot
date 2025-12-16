from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


documents = load_pdf_file(data=DATA_PATH)
print("number of pages in PDF: ", len(documents))

# Step 2: Create Chunks
def create_chunk(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunk(extracted_data=documents)
print("Length of text chunk: ", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_FAISS"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)