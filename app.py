import streamlit as st
import os
import time
import logging

# AI & LangChain Core
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
# These must be set in your Streamlit Cloud Dashboard (Settings > Secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DB_URL = st.secrets["DATABASE_URL"]

# Setup Page
st.set_page_config(page_title="Synapse AI", page_icon="⚡", layout="wide")
st.title("⚡ Synapse AI: Cloud Assistant")

# Log multi-query expansion in the app console
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- 2. SYSTEM INITIALIZATION ---
@st.cache_resource
def init_synapse():
    # Load Embeddings (CPU only for Cloud efficiency)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load Groq Llama 3.3
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=GROQ_API_KEY,
        temperature=0.2
    )
    
    # Initialize PGVector Store
    vector_store = PGVector(
        connection=DB_URL, 
        embeddings=embeddings, 
        collection_name="synapse_cloud_v1",
        use_jsonb=True
    )
    
    # --- CRITICAL DATABASE FIX ---
    # This creates the necessary SQL tables if your Neon DB is empty
    try:
        vector_store.create_tables_if_not_exists()
    except Exception as e:
        st.error(f"Database Initialization Error: {e}")
        
    return llm, vector_store

# Start the Brain
llm, vector_store = init_synapse()

# Setup Session History
if "history" not in st.session_state:
    st.session_state.history = []

# --- 3. SIDEBAR: PDF SYNC ---
with st.sidebar:
    st.header("📚 Synapse Memory")
    uploaded_file = st.file_uploader("Upload a PDF to Cloud DB", type="pdf")
    
    if uploaded_file and st.button("Sync with Neon DB"):
        with st.spinner("Chunking & Vectorizing..."):
            with open("temp_cloud.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Use PyPDF to read and split
            loader = PyPDFLoader("temp_cloud.pdf")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(loader.load())
            
            # Upload to Cloud Database
            vector_store.add_documents(docs)
            os.remove("temp_cloud.pdf")
            st.success(f"Added {len(docs)} chunks to Cloud Memory!")

    if st.button("🗑️ Wipe All Memory"):
        vector_store.delete_collection()
        st.session_state.history = []
        st.rerun()

# --- 4. ADVANCED RAG BRAIN ---

# Multi-Query: Expands 1 question into 3 to find more info
mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
    llm=llm
)

# Context Rephrasing: Handles chat history (pronouns like "it", "he")
context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the history, rephrase the question to be a standalone query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_retriever = create_history_aware_retriever(llm, mq_retriever, context_prompt)

# Organized QA Prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Synapse. Provide a **well-organized** answer based on context.
    - Use bullet points for lists.
    - Use ### Headings for sections.
    - Bold important dates/names.
    - If info is missing, say 'I couldn't find that in the document.'
    
    Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the final execution chain
rag_chain = create_retrieval_chain(history_retriever, create_stuff_documents_chain(llm, qa_prompt))

# --- 5. CHAT INTERFACE ---
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Ask Synapse anything..."):
    st.chat_message("user").markdown(user_input)
    
    with st.spinner("Retrieving from Cloud DB..."):
        try:
            start_time = time.time()
            response = rag_chain.invoke({
                "input": user_input, 
                "chat_history": st.session_state.history
            })
            end_time = time.time()
            
            answer = response["answer"]
            sources = response.get("context", [])
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Expandable Citation Box
                if sources:
                    with st.expander("🔍 View Synapse Sources"):
                        for i, doc in enumerate(sources):
                            page = doc.metadata.get("page", "N/A")
                            st.info(f"**Source {i+1} (Page {page}):**\n\n{doc.page_content[:300]}...")
                
                st.caption(f"⏱️ Response time: {round(end_time - start_time, 2)}s")
            
            # Save history
            st.session_state.history.extend([
                HumanMessage(content=user_input), 
                AIMessage(content=answer)
            ])
        except Exception as e:
            st.error(f"Chain Error: {e}")