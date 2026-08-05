from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader    
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

from streamlit import streamlit as st
from time import sleep


llm=ChatGoogleGenerativeAI(model='gemini-3.5-flash')

if "vector_db" not in st.session_state:
    st.session_state.vector_db=None

if "messages" not in st.session_state:
    st.session_state.messages=[]

def document_process(path):
   
##document loading
  loader=PyPDFLoader(path)
  docs=loader.load()
# print(f"Number of documents loaded: {len(docs)}")

##splitting the documents into chunks
  splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
  docs=splitter.split_documents(docs)

# print(f"Number of chunks created: {len(docs)}")

##embedding the documents and storing them in a vector store
  embeddings=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview'
)
  vector_db=InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=embeddings)
  st.session_state.vector_db=vector_db
  st.session_state.document_uploaded=True

##user query
# query="this is whose cv ? can this person built a tec start up in future "
# documents=vector_db.similarity_search(query=query,k=2)
# print(f"Number of documents retrieved: {len(documents)},{len(documents[0].page_content)},{documents[0].page_content}")

# context=""
# for doc in documents:
#     context=context+doc.page_content+'\n\n'

# prompt=f"""
# You are a helpful assistant that answers questions based on the context provided.
# Context:{context},question:{query}"""

 

# answer=llm.invoke(prompt)
# print(f"Answer: {answer.text}")


#ui section

st.subheader("📄Document based QnA Chatbot - Ask Anything") 
if 'document_uploaded' not in st.session_state:
  st.session_state.document_uploaded=False

### document upload section
if not st.session_state.document_uploaded:
    file=st.file_uploader(label='select your pdf file',type='pdf')

    if file:
       with open("uploaded_document.pdf",'wb') as f:
          f.write(file.getvalue())

       with st.spinner("Processing...."): #show a loader
         document_process('./uploaded_document.pdf')
       st.markdown("Document Uploaded sucessfully")  
       sleep(2)
       st.rerun()  #rerun the app to show the chat ui after document upload 


        

### chat ui    
if st.session_state.document_uploaded and st.session_state.vector_db:
   for oneMessage in st.session_state.messages:
      role=oneMessage["role"]
      content=oneMessage["content"]

      st.chat_message(role).markdown(content)

   query=st.chat_input("Ask Anything....")
   if query:
      

      st.session_state.messages.append({"role":"user","content":query})

      st.chat_message("user").markdown(query)
      documents=st.session_state.vector_db.similarity_search(query=query,k=2)
      context=" "

      for doc in documents:
             context=context+doc.page_content+'\n\n'

      prompt=f"""
You are a helpful assistant that answers questions based on the context provided.
Context:{context},question:{query}""" 
      answer=llm.invoke(prompt)
      st.session_state.messages.append({"role":"ai","content":answer.text})
     
      st.chat_message("ai").markdown(answer.text)





