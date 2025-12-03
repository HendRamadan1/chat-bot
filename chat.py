import streamlit as st
from streamlit_chat import message

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
# CHANGED: Import CSVLoader instead of PyPDFLoader

from langchain_classic.document_loaders.csv_loader import CSVLoader
import os
# REMOVED: Removed the need for tempfile
# import tempfile

# Initialisation functions (unchanged, but updated messages)
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        # UPDATED: Initial message for bank context
        st.session_state['generated'] = ["Hello! Ask me anything about the Bank's FAQs."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            # UPDATED: Placeholder for bank context
                    with st.form(key="form"):
                        user_input = st.text_input(
                            "Question:",
                            placeholder="Ask about bank FAQs (e.g., 'What should I do if my card is lost?')",
                            key='input'
                        )

                        submit_button = st.form_submit_button(label='Send')



        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create llm
    # NOTE: Ensure the mistral-7b-instruct-v0.1.Q4_K_M.gguf file is in the same directory
    # or accessible via the model_path.
    llm = LlamaCpp(
        streaming = True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    # UPDATED: Title to reflect the bank context
    st.title("Bank FAQs ChatBot using Mistral-7B-Instruct :bank:")

    # CHANGED: Removed file uploader and replaced with direct file loading
    st.sidebar.title("Document Processing")
    st.sidebar.markdown("Data loaded from **BankFAQs.csv**.") # Information for the user

    # Define the path to your CSV file
    csv_file_path =  "data/BankFAQs.csv" 

    # Use LangChain's CSVLoader to load the data
    # NOTE: The default setting will create document content from the entire row.
    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
    text = loader.load()

    # Text splitting remains the same
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings (remains the same)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Create vector store (remains the same)
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object (remains the same)
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
