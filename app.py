import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. SECRETS & CONFIG ---
# These are retrieved from st.secrets on the cloud dashboard
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_URL = st.secrets["DATABASE_URL"]

st.set_page_config(page_title="Synapse Cloud", page_icon="⚡", layout="wide")
st.title("⚡ Synapse AI: Cloud Knowledge Assistant")

# --- 2. MODEL INITIALIZATION ---
@st.cache_resource
def init_system():
    # Embeddings (Local CPU inference in Cloud)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    # LLM (Groq Llama 3.3)
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    
    # PGVector Store
    vector_store = PGVector(
        connection=DB_URL, 
        embeddings=embeddings, 
        collection_name="synapse_v1"
    )
    return llm, vector_store

llm, vector_store = init_system()

if "history" not in st.session_state:
    st.session_state.history = []

# --- 3. DATA INGESTION ---
with st.sidebar:
    st.header("📚 Document Sync")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.button("Index to Cloud"):
        with st.spinner("Processing..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(loader.load())
            vector_store.add_documents(docs)
            os.remove("temp.pdf")
            st.success("Successfully Synapsed!")

    if st.button("🗑️ Reset All"):
        vector_store.delete_collection()
        st.session_state.history = []
        st.rerun()

# --- 4. RAG BRAIN (Multi-Query Expansion) ---
mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
    llm=llm
)

context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the question to be a standalone query based on chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_retriever = create_history_aware_retriever(llm, mq_retriever, context_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Synapse. Use the context to answer professionally.
    - Use bullet points.
    - Bold key dates/names.
    - Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))

# --- 5. CHAT UI ---
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Ask Synapse..."):
    st.chat_message("user").markdown(user_input)
    
    with st.spinner("Thinking..."):
        start_time = time.time()
        response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.history})
        end_time = time.time()
        
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            if response.get("context"):
                with st.expander("🔍 View Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.info(f"Source {i+1} (Page {doc.metadata.get('page','')}): {doc.page_content[:200]}...")
            st.caption(f"⏱️ {round(end_time - start_time, 2)}s")
        
        st.session_state.history.extend([HumanMessage(content=user_input), AIMessage(content=response["answer"])])