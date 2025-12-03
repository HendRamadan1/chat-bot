# This is a Streamlit RAG application using ChromaDB for retrieval and 
# Mistral-7B-Instruct-v0.3 via the HuggingFace Inference API for generation.

# --- 1. Database Fix (Must be at the very top for Streamlit deployment) ---
# This ensures ChromaDB can use a modern SQLite version.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os

# Standard LangChain packages
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA # Corrected import path
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Removed unnecessary imports:
# HuggingFacePipeline, HuggingFaceHub, transformers, AutoTokenizer, torch

# --- 3. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# Constants
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "data/BankFAQs.csv" 

# Secrets Handling
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    # Use Streamlit secrets if available
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    # Fallback to user input for the API key
    api_key = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar or Streamlit Secrets.")
        st.stop()

# --- 4. Cached Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    """Loads CSV data, processes it into LangChain Documents, and creates/loads a Chroma vector store."""
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}. Please check your GitHub folder structure.")
        return None

    # Load and process the FAQ data
    bank = pd.read_csv(DATA_PATH)
    bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
    
    documents = []
    for _, row in bank.iterrows():
        documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

    # Initialize the embedding model
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the vector store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr"
    )
    return vector_db

@st.cache_resource
def load_llm():
    """Initializes the LLM using the HuggingFaceEndpoint wrapper for the Inference API."""
    # FIX: Using HuggingFaceEndpoint (API call) instead of loading the model locally.
    # FIX: All generation parameters (max_new_tokens, temperature, etc.) are passed directly
    # to resolve the Pydantic ValidationError.
    
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        max_new_tokens=512,         # Direct Argument
        do_sample=True,             # Direct Argument
        temperature=0.7,            # Direct Argument
        repetition_penalty=1.1,     # Direct Argument
    )
    return llm

# --- 5. App Logic ---

with st.spinner("Initializing AI Brain..."):
    vector_db = load_data_and_vectordb()
    # Check if API token is set before attempting to load LLM
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ and os.environ["HUGGINGFACEHUB_API_TOKEN"]:
        llm = load_llm()
    else:
        st.warning("API Token is not set. Cannot initialize LLM.")
        llm = None

if vector_db is None or llm is None:
    st.stop()

# Prompt Template using Mistral's instruction format
template = """<|system|>
You are a helpful and intelligent Finance QNA Expert for Banque Masr. 
Use the following context to answer the user's question accurately. 
If the answer is not in the context, say "Sorry, I don't know that information." and do not make up facts.
</s>
<|user|>
Context: {context}

Question: {question}
</s>
<|assistant|>
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Retriever and RAG Chain setup
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about loans, cards, or accounts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the RAG chain
                response = qa_chain.invoke(prompt)
                result = response['result']
                
                # Clean up tokens generated by the LLM
                if "<|assistant|>" in result:
                    result = result.split("<|assistant|>")[-1].strip()
                if "<|user|>" in result:
                    result = result.split("<|user|>")[0].strip()

                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                # Added robust error logging and display
                error_message = f"An error occurred during generation. This may be due to a malformed prompt or API issue: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
