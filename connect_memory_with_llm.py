from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


# --------------------------------------------------
# LLM (OPENAI)
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a medical information assistant.

You must answer the user's question using ONLY the information provided in the context.
Do NOT use any outside knowledge.
Do NOT make assumptions.
Do NOT generalize.

If the context does NOT contain enough information to answer the question, reply EXACTLY with:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""")

# --------------------------------------------------
# EMBEDDINGS + FAISS
# --------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_FAISS_PATH = "vectorstore/db_FAISS"

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# --------------------------------------------------
# LCEL RAG CHAIN
# --------------------------------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# QUERY LOOP
# --------------------------------------------------
while True:
    query = input("\nWrite Query Here (or 'exit'): ")
    if query.lower() == "exit":
        break

    # 🔍 Step 1: Inspect retrieved documents
    docs = retriever.invoke(query)

    print("\n🔍 Retrieved context from FAISS:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content)
        print("----------------\n")

    # 🤖 Step 2: Ask the model
    result = rag_chain.invoke(query)
    print("\nAnswer:\n", result)
