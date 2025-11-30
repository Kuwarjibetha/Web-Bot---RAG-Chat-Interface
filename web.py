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
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration - matching Gemini_web.py
client = genai.Client(api_key="Enter your Api Key")
os.environ["GOOGLE_API_KEY"] = "Enter your Api Key"

# This must be the first Streamlit command
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
if "base_url" not in st.session_state:
    st.session_state.base_url = None
if "available_urls" not in st.session_state:
    st.session_state.available_urls = []
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = set()
if "splitter" not in st.session_state:
    st.session_state.splitter = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "prompt" not in st.session_state:
    st.session_state.prompt = None

# Function to extract all internal links from a website
def extract_internal_links(base_url, max_pages=50):
    """
    Extract all internal links from a website starting from base_url.
    Returns a list of URLs to crawl.
    """
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    
    visited = set()
    to_visit = deque([base_url])
    all_urls = [base_url]
    
    try:
        while to_visit and len(all_urls) < max_pages:
            current_url = to_visit.popleft()
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            try:
                response = requests.get(current_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(current_url, href)
                    parsed = urlparse(absolute_url)
                    
                    # Only include internal links (same domain)
                    if parsed.netloc == parsed_base.netloc:
                        if absolute_url not in visited and absolute_url not in to_visit:
                            # Filter out common non-content URLs
                            if not any(exclude in absolute_url.lower() for exclude in [
                                '.pdf', '.jpg', '.png', '.gif', '.zip', '.exe', 
                                'mailto:', 'tel:', '#', 'javascript:', 'login', 'logout',
                                'signup', 'register', 'cart', 'checkout'
                            ]):
                                to_visit.append(absolute_url)
                                all_urls.append(absolute_url)
                                
            except Exception as e:
                st.warning(f"Could not process {current_url}: {str(e)}")
                continue
                
    except Exception as e:
        st.warning(f"Error during link extraction: {str(e)}")
    
    return list(set(all_urls))[:max_pages]  # Remove duplicates and limit

# Format docs functions - matching Gemini_web.py exactly
def format_docs(retrieved_docs):
    # Clean up each page content: replace newlines with spaces and remove bullet points
    cleaned_content = [doc.page_content.replace('\n', ' ').replace("   ","") for doc in retrieved_docs]
    context_text = " ".join(cleaned_content)
    return context_text

def format_docs_for_chain(retrieved_docs):
    # Format for RAG chain
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Function to determine if question needs navigation to another page
def needs_page_navigation(question, loaded_urls, available_urls, llm):
    """
    Use LLM to determine if the question requires data from a different page.
    Returns the URL to navigate to, or None if current data is sufficient.
    """
    if not available_urls or len(available_urls) <= 1:
        return None
    
    # Create a prompt to analyze if navigation is needed
    navigation_prompt = f"""Analyze this question and determine if it requires information from a specific page/section of the website.

Question: "{question}"

Available pages on the website:
{chr(10).join([f"- {url}" for url in available_urls[:20]])}

Already loaded pages:
{chr(10).join([f"- {url}" for url in list(loaded_urls)[:10]])}

Based on the question, determine:
1. Does the question ask about a specific section (like "career", "about", "services", "contact", "products", etc.)?
2. If yes, which URL from the available pages likely contains this information?
3. If the information might be in already loaded pages, return "NO_NAVIGATION"

Respond with ONLY one of:
- The exact URL to navigate to (if navigation is needed)
- "NO_NAVIGATION" (if current data is sufficient or question is general)

Response:"""
    
    try:
        response = llm.invoke(navigation_prompt).content.strip()
        
        if "NO_NAVIGATION" in response.upper() or "no navigation" in response.lower():
            return None
        
        # Try to extract URL from response
        for url in available_urls:
            if url in response:
                return url
        
        # If no exact match, try to find partial matches
        question_lower = question.lower()
        for url in available_urls:
            url_lower = url.lower()
            # Check if question mentions section names that might be in URL
            if any(keyword in question_lower and keyword in url_lower for keyword in 
                   ['career', 'about', 'service', 'product', 'contact', 'blog', 'news', 'team']):
                return url
        
        return None
    except:
        return None

# Function to load a new page and merge into existing vector DB
def load_and_merge_page(page_url, vector_db, splitter, embeddings):
    """
    Load a new page and merge its content into the existing vector database.
    Returns updated vector_db.
    """
    try:
        loader = WebBaseLoader(page_url)
        page_docs = loader.load()
        
        if not page_docs:
            return vector_db, False
        
        # Format and split the new documents
        web_word = format_docs(page_docs)
        new_chunks = splitter.create_documents([web_word])
        
        # Create temporary vector DB for new chunks
        temp_db = FAISS.from_documents(new_chunks, embeddings)
        
        # Merge with existing vector DB
        if vector_db is None:
            return temp_db, True
        else:
            vector_db.merge_from(temp_db)
            return vector_db, True
    except Exception as e:
        return vector_db, False

def need_rag(question):
    # Matching Gemini_web.py logic exactly
    q = question.lower()

    # If question is small talk or math or generic → no RAG
    general_keywords = ["hi", "hello", "hey", "who are you", "how are you", 
                        "thanks", "thank you", "bye", "good morning","ok"]

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
st.header(" Load a Webpage")
user_input = st.text_input("Enter URL", placeholder="https://example.com")
submit_button = st.button("Load URL")

if submit_button:
    if user_input and user_input.strip():
        url = user_input.strip()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Loading homepage and discovering website structure..."):
            try:
                # Step 1: Load homepage first (fast)
                status_text.text(" Loading homepage...")
                progress_bar.progress(20)
                
                loader = WebBaseLoader(url)
                homepage_docs = loader.load()
                
                if not homepage_docs:
                    st.error(" Could not load the homepage. Please check the URL.")
                    st.stop()
                
                # Step 2: Discover all available pages (but don't load them yet)
                status_text.text("Discovering all pages on the website...")
                progress_bar.progress(40)
                
                available_urls = extract_internal_links(url, max_pages=50)
                st.info(f"Found {len(available_urls)} pages available. Homepage loaded. Other pages will load automatically when needed.")
                
                # Step 3: Process and index homepage
                status_text.text("Processing and indexing homepage...")
                progress_bar.progress(60)
                
                # Clean up homepage content
                web_word = format_docs(homepage_docs)
                
                # Text Splitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([web_word])
                
                progress_bar.progress(80)
                
                # Embedding
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                vector_db = FAISS.from_documents(chunks, embeddings)
                
                progress_bar.progress(90)
                
                # Create retriever
                retriever = vector_db.as_retriever(search_kwargs={"k": 6})
                
                # Create prompt template
                prompt = PromptTemplate(
                    template="""
You are a helpful AI assistant with access to comprehensive information from all sections of a website.

Use the provided context from the website to answer questions accurately.
If the context contains information from multiple sections (like homepage, career page, about page, etc.), 
you can navigate between them to provide complete answers.

If useful context is provided, use it to answer.
If the context is not relevant or is empty, answer using your general knowledge.

----- CONTEXT FROM WEBSITE (may include multiple sections) -----
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
                st.session_state.prompt = prompt
                st.session_state.url_loaded = True
                st.session_state.messages = []  # Reset chat history when new URL is loaded
                st.session_state.base_url = url
                st.session_state.available_urls = available_urls
                st.session_state.loaded_urls = {url}  # Track loaded URLs
                st.session_state.splitter = splitter
                st.session_state.embeddings = embeddings
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Homepage loaded! Ask questions and I'll automatically load other pages when needed.")
                st.caption(f"💡 Try asking: 'What's on the career page?' or 'Tell me about the services section' - I'll load those pages automatically!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error loading URL: {str(e)}")
    else:
        st.error("Please enter a valid URL")

# Chat interface - matching Gemini_web.py while loop logic
if st.session_state.url_loaded:
    st.divider()
    st.header("Ask Questions")
    loaded_count = len(st.session_state.loaded_urls)
    st.caption(f"Ask questions about any section. Currently loaded: {loaded_count} page(s). I'll automatically load other pages when needed. Type 'exit', 'quit', or 'goodbye' to end.")
    
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
                        # Step 1: Check if navigation to another page is needed
                        if need_rag(question):
                            target_url = needs_page_navigation(
                                question, 
                                st.session_state.loaded_urls,
                                st.session_state.available_urls,
                                st.session_state.llm
                            )
                            
                            # If navigation is needed and URL not already loaded
                            if target_url and target_url not in st.session_state.loaded_urls:
                                with st.spinner(f"🔍 Loading page: {target_url}..."):
                                    updated_db, success = load_and_merge_page(
                                        target_url,
                                        st.session_state.vector_db,
                                        st.session_state.splitter,
                                        st.session_state.embeddings
                                    )
                                    
                                    if success:
                                        st.session_state.vector_db = updated_db
                                        st.session_state.loaded_urls.add(target_url)
                                        
                                        # Recreate retriever and chain with updated vector DB
                                        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 6})
                                        parallel_chain = RunnableParallel({
                                            'context': retriever | RunnableLambda(format_docs_for_chain),
                                            'question': RunnablePassthrough()
                                        })
                                        parser = StrOutputParser()
                                        st.session_state.main_chain = parallel_chain | st.session_state.prompt | st.session_state.llm | parser
                                        
                                        st.info(f"✅ Loaded new page: {target_url}")
                            
                            # Use RAG chain (context + LLM)
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
