# --- 1. Database Fix (Must be at the very top) ---
# Required for running Chroma DB on some environments (like Streamlit Cloud)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # Use standard sqlite3 if available

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 3. Banque Masr Static Data (Replacing BankFAQs.csv) ---
# Embedding the data directly for portability since file access is not guaranteed.
BANK_FAQS = [
    {"question": "What are the requirements for opening a new savings account?", "answer": "You need a valid national ID, a recent utility bill, and an initial deposit of 1000 EGP.", "class": "Accounts"},
    {"question": "What is the maximum duration for a personal loan?", "answer": "The maximum duration is 7 years, or 84 months, subject to credit score approval.", "class": "Loans"},
    {"question": "How do I report a lost or stolen credit card?", "answer": "Immediately call our 24/7 hotline at 19666. Your card will be instantly blocked. This is a critical security measure.", "class": "Cards"},
    {"question": "Can I apply for a mortgage if I am self-employed?", "answer": "Yes, provided you can show consistent income statements for the past two years and provide business registration documents.", "class": "Loans"},
    {"question": "What are the monthly maintenance fees for the Platinum account?", "answer": "The monthly maintenance fee is 50 EGP, which is waived if the minimum balance of 20,000 EGP is maintained for the entire month.", "class": "Accounts"},
    {"question": "Do you offer car loans?", "answer": "Yes, car loans are available for both new and used vehicles, with repayment terms up to 5 years.", "class": "Loans"},
]
# Constants
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"


# --- 4. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# Secrets Handling
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    api_key = st.sidebar.text_input("Enter Hugging Face API Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar.")
        st.stop()


# --- 5. Cached Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    st.write("Preparing knowledge base...")
    
    # Process embedded data into LangChain Documents
    documents = []
    for faq in BANK_FAQS:
        content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        documents.append(Document(page_content=content, metadata={"class": faq["class"]}))

    # Create embeddings using the same model as your original code
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store (Chroma)
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr"
    )
    st.write("Knowledge base ready.")
    return vector_db

@st.cache_resource
def load_llm():
    st.write("Loading LLM...")
    # Initialize HuggingFace Endpoint LLM
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        task="text-generation" 
    )
    st.write("LLM loaded.")
    return llm

# --- 6. Chain Setup ---

# Prompt Template (Mistral formatting)
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

# Load resources
try:
    vector_db = load_data_and_vectordb()
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize RAG components. Check API token and dependencies: {e}")
    st.stop()

# Initialize Retriever and QA Chain
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


# --- 7. Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about loans, cards, or accounts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FAQs..."):
            try:
                # Invoke the RAG chain
                response = qa_chain.invoke(prompt)
                result = response.get('result', "I'm having trouble retrieving an answer right now.")
                
                # Cleanup Mistral-specific tokens that may appear in the output
                if "<|assistant|>" in result:
                    result = result.split("<|assistant|>")[-1].strip()
                
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"An error occurred during API call: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "An internal error occurred. Please check the logs."})
