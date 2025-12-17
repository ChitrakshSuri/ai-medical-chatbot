import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_FAISS"

# --------------------------------------------------
# CACHE: VECTORSTORE + RAG CHAIN
# --------------------------------------------------
@st.cache_resource
def load_rag_chain():
    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5
    )

    # Prompt
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

    # Embeddings + FAISS
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LCEL RAG chain
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------
def main():
    st.title("🩺 Medical RAG Chatbot")

    rag_chain, retriever = load_rag_chain()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a medical question")

    if user_query:
        # User message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        # (Optional) show retrieved chunks
        with st.expander("🔍 Retrieved context"):
            docs = retriever.invoke(user_query)
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(d.page_content)

        # Assistant response
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_query)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
