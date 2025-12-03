# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 3. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# Constants
# CHANGED: Switched to Flan-T5. This is a text-to-text model that bypasses
# the "Conversational" task restriction error you were facing.
REPO_ID = "google/flan-t5-large"
DATA_PATH = "data/BankFAQs.csv" 

# Secrets Handling
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    api_key = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar or Streamlit Secrets.")
        st.stop()

# --- 4. Cached Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}. Please check your GitHub folder structure.")
        return None

    bank = pd.read_csv(DATA_PATH)
    # Combine question and answer for better context retrieval
    bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
    
    documents = []
    for _, row in bank.iterrows():
        documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

    # Setup Embeddings
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Setup Vector DB
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr"
    )
    return vector_db

@st.cache_resource
def load_llm():
    # Flan-T5 works best with these parameters
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="text2text-generation",
        max_new_tokens=512,
        do_sample=False, # T5 works better with greedy search for QA
        temperature=0.1
    )
    return llm

# --- 5. App Logic ---

with st.spinner("Initializing AI Brain..."):
    vector_db = load_data_and_vectordb()
    llm = load_llm()

if vector_db is None or llm is None:
    st.stop()

# Prompt Template
# CHANGED: Standard RAG prompt for T5 (no <|system|> tags needed)
template = """Use the following pieces of context to answer the question at the end. 
If the answer is not in the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

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
                response = qa_chain.invoke(prompt)
                result = response['result']
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {str(e)}")
