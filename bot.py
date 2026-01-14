import gradio as gr
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = "AIzaSyBhHXTdbe8E40zJ5g_aFqt3QJvTPDkqOAk"

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.5,
        max_output_tokens=256
    )
    return llm

def document_loader(file):
    loader = PyMuPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def get_embeddings():
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def vector_database(chunks):
    embedding_model = get_embeddings()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vector_db

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vector_db = vector_database(chunks)
    retriever_obj = vector_db.as_retriever()
    return retriever_obj

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever_obj,
        return_source_documents=True
    )
    response = qa.invoke({'query': query})
    return response['result']

rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode='never',
    inputs=[
        gr.File(label='Upload PDF', file_count='single', file_types=['.pdf']),
        gr.Textbox(label='Input Query', lines=2, placeholder='Type your Question...')
    ],
    outputs=gr.TextArea(label='Answer'),
    title='QA BOT',
    description='Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.'
)

if __name__ == "__main__":
    rag_application.launch(server_name='127.0.0.1', server_port=7682)