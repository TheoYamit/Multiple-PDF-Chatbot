import streamlit as sl
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

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
    
def main():
    load_dotenv()
    sl.set_page_config(page_title="PDF chat", page_icon="")
    sl.header("PDF chat")
    sl.text_input("Type your question regarding your uploaded pdf's...")

    with sl.sidebar:
        sl.subheader('Your uploaded :red[documents]', divider='grey')
        pdfs = sl.file_uploader("Upload multiple PDF's", accept_multiple_files=True)
        col1, col2, col3 = sl.columns(3)
        with col2:
            if sl.button("Process PDF's"):
                with sl.spinner("Processing PDF's..."): 
                    # Getting the text and splitting into chunks from pdfs
                    chunks = get_text_and_split(pdfs)
                    vectorestore = vector_store(chunks)
                    test = vectorestore.similarity_search("The existence of the shogunate rested on which two obligiations")
                    sl.write(test)
                    





if __name__ == '__main__':
    main()