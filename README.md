# Web Bot - Intelligent RAG Chat Interface

An intelligent web scraping and Q&A system that allows you to interact with websites through natural language. The system uses advanced AI to automatically navigate and extract information from websites, building knowledge incrementally as you ask questions.

## 🚀 Features

### Intelligent Page Navigation
- **Smart Loading**: Starts by loading only the homepage for fast initial access
- **On-Demand Navigation**: Automatically detects when questions require information from other pages and loads them in the background
- **Incremental Learning**: All loaded pages are merged into the knowledge base, so previous questions remain answerable
- **No Manual Navigation**: You don't need to specify which page to load - the AI decides automatically

### Advanced RAG (Retrieval-Augmented Generation)
- **Comprehensive Context**: Uses vector embeddings to retrieve relevant information from all loaded pages
- **Multi-Section Answers**: Can answer questions using information from multiple sections of a website
- **Context-Aware**: Understands when to use web content vs general knowledge

### User-Friendly Interface
- **Streamlit Web Interface**: Clean and intuitive chat interface
- **Real-Time Loading**: Visual progress indicators during page loading
- **Chat History**: Maintains conversation context throughout the session

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Internet connection for web scraping

## 🔧 Installation

1. **Clone or download this repository**

2. **Install required packages**:
```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install google-generativeai
pip install beautifulsoup4
pip install requests
pip install faiss-cpu  # or faiss-gpu for GPU support
```

Or install all at once:
```bash
pip install streamlit langchain langchain-community langchain-google-genai google-generativeai beautifulsoup4 requests faiss-cpu
```

3. **Configure API Key**:
   - Open `web.py`
   - Replace `"AIzaSyDTYFlukvgiQBF8aQbDfHl1wMWU-G7gcRc"` with your Google Gemini API key (lines 19-20)
   - You can get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🎯 Usage

### Starting the Application

```bash
streamlit run web.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Web Bot

#### Step 1: Load a Webpage
1. Enter a website URL in the input field (e.g., `https://example.com`)
2. Click "Load URL"
3. The system will:
   - Load the homepage
   - Discover all available pages on the website
   - Index the homepage content

#### Step 2: Ask Questions
1. Type your question in the chat input
2. Examples:
   - "What's on the career page?"
   - "Tell me about the services section"
   - "What does the about page say?"
   - "Summarize the homepage"

#### How It Works

1. **Initial Load**: Only the homepage is loaded (fast!)
2. **Smart Detection**: When you ask about a specific section, the AI analyzes your question
3. **Automatic Navigation**: If needed, the relevant page is loaded automatically
4. **Data Merging**: New page content is added to existing knowledge (nothing is lost!)
5. **Comprehensive Answer**: You get an answer using information from all relevant pages

## 🏗️ Architecture

### Technology Stack

- **Frontend**: Streamlit - Interactive web interface
- **LLM**: Google Gemini 2.0 Flash - Advanced language model
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS - Fast similarity search
- **Web Scraping**: 
  - BeautifulSoup - HTML parsing
  - WebBaseLoader (LangChain) - Document loading
- **Text Processing**: RecursiveCharacterTextSplitter - Chunking for embeddings

### Key Components

1. **Link Extractor** (`extract_internal_links`): Discovers all internal links on a website
2. **Page Loader** (`load_and_merge_page`): Loads and merges new pages into the vector database
3. **Navigation Detector** (`needs_page_navigation`): Uses AI to determine if a question requires loading another page
4. **RAG Chain**: Combines retrieval and generation for accurate, context-aware answers

## 💡 Example Workflow

```
User: Enter URL → https://company.com

System: 
  ✓ Loading homepage...
  ✓ Discovering all pages on the website...
  ✓ Found 25 pages available
  ✓ Homepage loaded!

User: "What does the company do?"

System: [Answers using homepage content]

User: "What jobs are available?"

System: 
  ✓ Loading page: https://company.com/careers
  [Answers using career page content]

User: "What does the company do?" [Same question again]

System: [Still works! All data is preserved]
```

## 🔍 How It Differs from Simple Web Scraping

- **Intelligent**: Decides when to load additional pages automatically
- **Incremental**: Only loads what's needed, when it's needed
- **Persistent**: All loaded data accumulates in memory
- **Context-Aware**: Understands question intent and navigates accordingly
- **Fast Initial Load**: Homepage loads quickly; other pages load on-demand

## ⚙️ Configuration

### Adjusting Crawl Limits

In `web.py`, you can modify:
- `max_pages=50` in `extract_internal_links()` - Maximum pages to discover
- `search_kwargs={"k": 6}` - Number of document chunks to retrieve for context
- `chunk_size=1000, chunk_overlap=200` - Text chunking parameters

### Model Settings

- **Model**: `gemini-2.0-flash` (can be changed in line 305)
- **Temperature**: `0.2` (lower = more deterministic)

## 🐛 Troubleshooting

### Common Issues

1. **"Could not load the homepage"**
   - Check if the URL is correct and accessible
   - Some websites may block automated access

2. **Slow loading**
   - Large websites take longer to discover links
   - Consider reducing `max_pages` parameter

3. **API Key errors**
   - Verify your Google Gemini API key is correct
   - Check if you have API quota remaining

4. **Missing pages**
   - Some pages may require authentication
   - JavaScript-rendered content may not be accessible

## 📝 Notes

- The system filters out non-content URLs (PDFs, images, login pages, etc.)
- Only internal links (same domain) are processed
- Chat history persists during the session
- Loading a new URL resets the chat history

## 🔐 Security

- API keys are stored in the code (consider using environment variables for production)
- Only publicly accessible web pages can be loaded
- No data is stored permanently - all data is in session memory

## 📄 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For questions or issues, please check the code comments or open an issue in the repository.

---

**Enjoy chatting with websites! 🎉**

