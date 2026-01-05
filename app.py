import streamlit as st
import os
import time
import logging

# AI & LangChain Core (2026 Modular Imports)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Advanced RAG Components
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CLOUD CONFIGURATION ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_URL = st.secrets["DATABASE_URL"]
ADMIN_PASS = st.secrets["ADMIN_PASSWORD"]

# Page Setup
st.set_page_config(page_title="Synapse AI", page_icon="⚡", layout="wide")
st.title("⚡ Synapse AI: Enterprise RAG")

# --- 2. SYSTEM INITIALIZATION ---
@st.cache_resource
def init_synapse():
    # Load Embeddings (CPU Optimized)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load LLM (llama-3.1-8b for TPD efficiency and speed)
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        temperature=0.0 # Deterministic for factual accuracy
    )
    
    # Initialize Vector Store with SSL & Connection Pooling
    vector_store = PGVector(
        connection=DB_URL, 
        embeddings=embeddings, 
        collection_name="synapse_production_v1",
        use_jsonb=True,
        engine_args={
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }
    )
    
    # Provision Database Tables
    try:
        vector_store.create_tables_if_not_exists()
    except Exception as e:
        st.error(f"Database Initialization Error: {e}")
        
    return llm, vector_store

# Launch Brain
llm, vector_store = init_synapse()

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# --- 3. SECURE SIDEBAR ---
with st.sidebar:
    st.header("🔐 Admin Panel")
    password = st.text_input("Admin Password", type="password")
    
    if password == ADMIN_PASS:
        st.success("Authorized")
        st.divider()
        st.header("📚 Knowledge Manager")
        
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and st.button("Sync to Cloud DB"):
            with st.spinner("Indexing for High-Precision Retrieval..."):
                temp_file = f"temp_{int(time.time())}.pdf"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Optimized Load & Split
                    loader = PyPDFLoader(temp_file)
                    raw_pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    
                    # Fix for subscriptable error: Force list type
                    docs = text_splitter.split_documents(list(raw_pages))
                    
                    # Add Metadata
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    vector_store.add_documents(docs)
                    st.success(f"Successfully Synapsed {len(docs)} chunks!")
                except Exception as e:
                    st.error(f"Sync Failed: {e}")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

        if st.button("🗑️ Reset All Knowledge"):
            vector_store.delete_collection()
            st.session_state.history = []
            st.rerun()
    else:
        if password: st.error("Access Denied")
        st.info("Knowledge indexing is restricted to admins.")

# --- 4. THE RAG BRAIN (Multi-Query Expansion) ---

# 1. Multi-Query Retriever
mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
    llm=llm
)

# 2. Context Re-phraser (Standalone Question Generator)
context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the history, rephrase the user's query into a standalone research question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_retriever = create_history_aware_retriever(llm, mq_retriever, context_prompt)

# 3. Final Answer Prompt (Strict Grounding)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Synapse. Your answers must be BASED ONLY on the provided context.
    - If the info is missing, say: "I couldn't find that in the documents."
    - Be professional and use bullet points for clarity.
    
    Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))

# --- 5. CHAT UI ---
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Query the knowledge base..."):
    st.chat_message("user").markdown(user_input)
    
    with st.spinner("Consulting Synapse Brain..."):
        try:
            start = time.time()
            response = rag_chain.invoke({
                "input": user_input, 
                "chat_history": st.session_state.history
            })
            
            answer = response["answer"]
            sources = response.get("context", [])
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                if sources:
                    with st.expander("🔍 Evidence & Sources"):
                        for i, doc in enumerate(sources):
                            src = doc.metadata.get("source", "Unknown")
                            pg = doc.metadata.get("page", "?")
                            st.caption(f"**Doc {i+1}:** {src} (Page {pg})")
                            st.write(f"{doc.page_content[:250]}...")
            
            st.session_state.history.extend([
                HumanMessage(content=user_input), 
                AIMessage(content=answer)
            ])
            st.caption(f"⚡ Latency: {round(time.time() - start, 2)}s")
        except Exception as e:
            st.error(f"Chain Execution Error: {e}")