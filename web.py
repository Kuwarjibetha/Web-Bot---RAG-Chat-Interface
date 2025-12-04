import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

client = genai.Client(api_key="API key")
os.environ["GOOGLE_API_KEY"] = "API key"


st.set_page_config(layout="wide")
st.title("Web Bot - RAG Chat Interface")


# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "url_loaded" not in st.session_state:
    st.session_state.url_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

#  docs functions 
def format_docs(retrieved_docs):
    # Clean up each page content: replace newlines with spaces and remove bullet points
    cleaned_content = [doc.page_content.replace('\n', ' ').replace("   ","") for doc in retrieved_docs]
    context_text = " ".join(cleaned_content)
    return context_text

def format_docs_for_chain(retrieved_docs):
    # Format for RAG chain
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def need_rag(question):
    # Matching Gemini_web.py logic exactly
    q = question.lower()

    # If question is small talk or math or generic → no RAG
    general_keywords = ["hi", "hello", "hey", "who are you", "how are you", 
                        "thanks", "thank you", "bye", "good morning"]

    if q in general_keywords:
        return False

    # If it's a simple numeric/math question → no RAG
    if any(char.isdigit() for char in q):
        return False

    # If question clearly asks for web page info → RAG
    rag_keywords = ["from the website", "according to the website", "in the site",
                    "in the article", "what does the page say", "web", "context"]

    if any(k in q for k in rag_keywords):
        return True

    # Default = ask both general + rag
    return True

# URL input section
st.header("Step 1: Load a Webpage")
user_input = st.text_input("Enter URL", placeholder="https://example.com")
submit_button = st.button("Load URL")

if submit_button:
    if user_input and user_input.strip():
        url = user_input.strip()
        with st.spinner("Loading and processing URL..."):
            try:
                # Load documents
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # Clean up each page content: replace newlines with spaces and remove bullet points
                web_word = format_docs(docs)
                
                # Text Splitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([web_word])
                
                # Embedding
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                vector_db = FAISS.from_documents(chunks, embeddings)
                
                # Create retriever
                retriever = vector_db.as_retriever(search_kwargs={"k": 4})
                
                # Create prompt template
                prompt = PromptTemplate(
                    template="""
You are a helpful AI assistant.

If useful context is provided, use it to answer.
If the context is not relevant or is empty, answer using your general knowledge.

----- CONTEXT (may be empty) -----
{context}
----------------------------------

Question: {question}

Reply:
    """,
                    input_variables=['context', 'question']
                )
                
                # Create LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.2
                )
                
                # Create parallel chain
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs_for_chain),
                    'question': RunnablePassthrough()
                })
                
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser
                
                # Store in session state
                st.session_state.vector_db = vector_db
                st.session_state.main_chain = main_chain
                st.session_state.llm = llm
                st.session_state.url_loaded = True
                st.session_state.messages = []  # Reset chat history when new URL is loaded
                
                st.success(f"✅ URL loaded successfully! You can now ask questions about the content.")
            except Exception as e:
                st.error(f"❌ Error loading URL: {str(e)}")
    else:
        st.error("Please enter a valid URL")

# Chat interface - matching Gemini_web.py while loop logic
if st.session_state.url_loaded:
    st.divider()
    st.header("Step 2: Ask Questions")
    st.caption("Ask questions about the loaded webpage. Type 'exit', 'quit', or 'goodbye' to end the conversation.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question..."):
        question = question.strip()
        
        # Check for exit commands - matching Gemini_web.py
        if question.lower() in ["exit", "quit", "goodbye"]:
            st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
            with st.chat_message("assistant"):
                st.markdown("Goodbye!")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response - matching Gemini_web.py logic
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Smart decision - matching Gemini_web.py
                        if need_rag(question):
                            # Use full RAG chain (context + LLM)
                            answer = st.session_state.main_chain.invoke(question)
                        else:
                            # Skip RAG → direct LLM answer (normal chat)
                            answer = st.session_state.llm.invoke(question).content
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("👆 Please load a URL first to start chatting about its content.")
