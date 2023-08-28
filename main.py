import streamlit as sl
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlCode import styles, human_class, robot_class, fade_in_css
import time
import sys

def get_text_and_split(pdfs):
    raw_text = ""
    for pdf_doc in pdfs:
        pdf_reader = PdfReader(pdf_doc)
        for page_num in pdf_reader.pages:
            text = page_num.extract_text()
            raw_text += text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len, 
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en')
    pinecone.init(
        api_key='b7540fd4-1c50-4982-ba4e-d38204d59337',
        environment='gcp-starter'
    )
    pinecone.Index('chatbotmultiplepdfs')

    vectorstore = Pinecone.from_texts(chunks, embeddings, index_name='chatbotmultiplepdfs')
    return vectorstore

def conversation(vectorstore):
    language_model = HuggingFaceHub(repo_id="HuggingFaceM4/idefics-9b-instruct", model_kwargs={"temperature":0.5, "max_length":4096})
    
    memory = ConversationBufferMemory(
        memory_key = 'chat_history',    
        return_messages=True,
    )

    conversation = ConversationalRetrievalChain.from_llm(   
        llm = language_model,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    return conversation

def answer_question(asked_question):
    answer = sl.session_state.chat_conversation({'question': asked_question})
    sl.session_state.chat_history = answer['chat_history']

    for i, message in enumerate(sl.session_state.chat_history):
        print(f"Message {i}: {message.content}\n")
        if i % 2 == 0:
            sl.write(human_class.replace(
                "{{human message}}", message.content), unsafe_allow_html=True)
        else:
            sl.write(robot_class.replace(
                "{{robot message}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    sl.set_page_config(page_title="PDF chat", page_icon="")

    sl.write(styles, unsafe_allow_html=True)

    if "chat_conversation" not in sl.session_state:
        sl.session_state.chat_conversation = None

    if "chat-history" not in sl.session_state:
        sl.session_state.chat_history = None

    sl.header("PDF chat")
    question = sl.text_input("Type your question regarding your uploaded pdf's...")

    sl.markdown(fade_in_css, unsafe_allow_html=True)
    sl.write(human_class.replace("{{human message}}", "Yo, robot! Please answer my questions on my pdfs!"), unsafe_allow_html=True)
    time.sleep(2)
    sl.write(robot_class.replace("{{robot message}}", "Hello, human. I will glady do so. Just upload your pdfs!"), unsafe_allow_html=True)

    if question:
        answer_question(question)

    with sl.sidebar:
        sl.subheader('Your uploaded :red[documents]', divider='grey')
        pdfs = sl.file_uploader("Upload multiple PDF's", accept_multiple_files=True)
        col1, col2, col3 = sl.columns(3)
        with col2:
            if sl.button("Process PDF's"):
                with sl.spinner("Processing PDF's..."): 
                    # Getting the text and splitting into chunks from pdfs
                    chunks = get_text_and_split(pdfs)
                    vectorstore = vector_store(chunks)
                   
                    sl.session_state.chat_conversation = conversation(vectorstore)
    

if __name__ == '__main__':
    main()