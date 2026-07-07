import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error(
        "⚠️ GOOGLE_API_KEY is not set. "
        "Add it as an environment variable in Render (Settings → Environment) "
        "or set it locally before running the app."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(layout="wide")
st.title("Web Bot - RAG Chat Interface")

# Session state initialization

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



# Helper functions

def format_docs(retrieved_docs):
    """Clean up page content: replace newlines with spaces, strip extra spaces."""
    cleaned_content = [
        doc.page_content.replace("\n", " ").replace("   ", "")
        for doc in retrieved_docs
    ]
    return " ".join(cleaned_content)


def format_docs_for_chain(retrieved_docs):
    """Format retrieved docs for the RAG chain."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def need_rag(question):
    """Decide whether the question needs website context (RAG) or general knowledge."""
    q = question.lower()

    general_keywords = [
        "hi", "hello", "hey", "who are you", "how are you",
        "thanks", "thank you", "bye", "good morning",
    ]
    if q in general_keywords:
        return False

    if any(char.isdigit() for char in q):
        return False

    rag_keywords = [
        "from the website", "according to the website", "in the site",
        "in the article", "what does the page say", "web", "context",
    ]
    if any(k in q for k in rag_keywords):
        return True

    return True



# Step 1: Load a webpage

st.header("Step 1: Load a Webpage")
user_input = st.text_input("Enter URL", placeholder="https://example.com")
submit_button = st.button("Load URL")

if submit_button:
    if user_input and user_input.strip():
        url = user_input.strip()
        with st.spinner("Loading and processing URL..."):
            try:
                # Load the page
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Clean text
                web_word = format_docs(docs)

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([web_word])

                # Embeddings + vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                vector_db = FAISS.from_documents(chunks, embeddings)

                # Retriever
                retriever = vector_db.as_retriever(search_kwargs={"k": 4})

                # Prompt template
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
                    input_variables=["context", "question"],
                )

                # LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.2,
                )

                # RAG chain
                parallel_chain = RunnableParallel(
                    {
                        "context": retriever | RunnableLambda(format_docs_for_chain),
                        "question": RunnablePassthrough(),
                    }
                )
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                # Save to session state
                st.session_state.vector_db = vector_db
                st.session_state.main_chain = main_chain
                st.session_state.llm = llm
                st.session_state.url_loaded = True
                st.session_state.messages = []  # reset chat on new URL

                st.success("✅ URL loaded successfully! You can now ask questions about the content.")
            except Exception as e:
                st.error(f"❌ Error loading URL: {str(e)}")
    else:
        st.error("Please enter a valid URL")


# Step 2: Chat interface
if st.session_state.url_loaded:
    st.divider()
    st.header("Step 2: Ask Questions")
    st.caption("Ask questions about the loaded webpage. Type 'exit', 'quit', or 'goodbye' to end the conversation.")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # New input
    if question := st.chat_input("Ask a question..."):
        question = question.strip()

        if question.lower() in ["exit", "quit", "goodbye"]:
            st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
            with st.chat_message("assistant"):
                st.markdown("Goodbye!")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if need_rag(question):
                            answer = st.session_state.main_chain.invoke(question)
                        else:
                            answer = st.session_state.llm.invoke(question).content

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("👆 Please load a URL first to start chatting about its content.")
