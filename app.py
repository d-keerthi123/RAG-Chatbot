import streamlit as st
import pdfplumber 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

st.header("RAG-Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")

# get user question
user_question=st.text_input("Type your question here")

# Extract the pdf and chunk it
if file is not None:
    # extract test from it
    
    with pdfplumber.open(file) as pdf:
        text=""
        for page in pdf.pages:
            text+=page.extract_text() + "\n"
    # st.write(text)

    # split data into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," ",""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(text)
    # st.write(chunks)

    # Generate Embeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # store embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embeddings)

    

    # generate answer
    # question -> embeddings -> similarity search -> results to llm -> response
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant answering questions about a PDF document.
         Guidelines:
         1. Provide complete, well-explained answers using the context below.
         2. Include relevant details, numbers, and explanations.
         3. If related information exists in context, include it.
         4. Only use information from the provided context.
         5. Summarize long answers when needed.
         6. If answer is not in the context, say you don't know politely.
         Context: \n{context}"""),
         ("human", "{question}")
         ])

    # LLM
    llm = Ollama(model="tinyllama")


    # Define format function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
    )

    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    if user_question:
        with st.spinner("Generating answer..."):
            response=chain.invoke(user_question)
            st.write(response)


