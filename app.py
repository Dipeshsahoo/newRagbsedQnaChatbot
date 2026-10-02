from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader    
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

import streamlit as st
from time import sleep

st.set_page_config(page_title="AI Document Analyst", page_icon="📄", layout="wide")

# Custom UI styles
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


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

if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False

# Sidebar for document upload
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=80)
    st.title("Document Upload")
    st.markdown("Upload your PDF to start asking questions.")
    
    if not st.session_state.document_uploaded:
        file = st.file_uploader(label="Select your PDF file", type='pdf', label_visibility="collapsed")
        
        if file:
            with open("uploaded_document.pdf", 'wb') as f:
                f.write(file.getvalue())
            
            with st.spinner("Processing Document... ⏳"):
                document_process('./uploaded_document.pdf')
            st.success("✅ Document Uploaded Successfully!")
            sleep(1.5)
            st.rerun()
    else:
        st.success("✅ Document is loaded and ready.")
        if st.button("Upload a different document", use_container_width=True):
            st.session_state.document_uploaded = False
            st.session_state.messages = []
            st.session_state.vector_db = None
            st.rerun()

# Main Chat Interface
st.title("📄 AI Document Analyst")
st.markdown("Ask anything about your document and get instant, accurate answers.")
st.divider()

if st.session_state.document_uploaded and st.session_state.vector_db:
    # Display chat history
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        st.chat_message(role, avatar=avatar).markdown(content)
        
    if len(st.session_state.messages) == 0:
        st.info("👋 Welcome! Your document is ready. Ask me your first question below!")

    query = st.chat_input("Ask anything about the document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user", avatar="🧑‍💻").markdown(query)
        
        with st.chat_message("ai", avatar="🤖"):
            with st.spinner("Analyzing..."):
                documents = st.session_state.vector_db.similarity_search(query=query, k=3)
                context = " ".join([doc.page_content for doc in documents])
                
                prompt = f"""
You are a highly capable AI assistant that answers questions accurately based on the provided context.
If the answer is not in the context, politely inform the user.
Context: {context}

Question: {query}
"""
                answer = llm.invoke(prompt)
                st.markdown(answer.text)
                st.session_state.messages.append({"role": "ai", "content": answer.text})
else:
    st.info("👈 Please upload a PDF document from the sidebar to begin.")
