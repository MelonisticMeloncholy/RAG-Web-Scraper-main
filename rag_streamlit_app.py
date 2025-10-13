import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
from urllib.parse import urlparse

# Load environment variables (for default API URL, though we'll hardcode here for clarity)
load_dotenv()

# --- Configuration ---
# Ensure this matches where your FastAPI service is running
BASE_API_URL = "http://127.0.0.1:8001" 
CRAWL_ENDPOINT = f"{BASE_API_URL}/crawl"
ASK_ENDPOINT = f"{BASE_API_URL}/ask"

# Default values for inputs
DEFAULT_CRAWL_URL = os.getenv("DEFAULT_CRAWL_URL", "https://seths.blog/")
DEFAULT_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", 5))
DEFAULT_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", 1))
DEFAULT_CRAWL_DELAY_MS = int(os.getenv("CRAWL_DELAY_MS", 200))
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="RAG Service Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 RAG Service Demo")
st.markdown("Interact with your Retrieval-Augmented Generation (RAG) service.")

# --- Session State Initialization ---
if 'crawl_status' not in st.session_state:
    st.session_state.crawl_status = {"message": "Service ready. Enter a URL to start crawling.", "success": None}
if 'crawled_urls' not in st.session_state:
    st.session_state.crawled_urls = []
if 'vector_count' not in st.session_state:
    st.session_state.vector_count = 0
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None


# --- Sidebar for Status and Controls ---
with st.sidebar:
    st.header("Service Status")
    if st.session_state.crawl_status["success"] is True:
        st.success(st.session_state.crawl_status["message"])
    elif st.session_state.crawl_status["success"] is False:
        st.error(st.session_state.crawl_status["message"])
    else:
        st.info(st.session_state.crawl_status["message"])
    
    st.metric("Indexed Pages", len(st.session_state.crawled_urls))
    st.metric("Indexed Vectors", st.session_state.vector_count)

    st.markdown("---")
    st.header("API Configuration")
    api_url_input = st.text_input("FastAPI Base URL", BASE_API_URL)
    if api_url_input != BASE_API_URL:
        BASE_API_URL = api_url_input
        CRAWL_ENDPOINT = f"{BASE_API_URL}/crawl"
        ASK_ENDPOINT = f"{BASE_API_URL}/ask"
        st.warning(f"API endpoints updated. Ensure your FastAPI service is running at {BASE_API_URL}")

    st.markdown("---")
    st.caption("Developed by Developed by Haresh for the RAG service project.")


# --- Main Content Area ---

# 1. Crawl Section
st.header("🌐 Crawl & Index Website")
st.write("Provide a starting URL to crawl and index its content for RAG queries.")

with st.form("crawl_form"):
    start_url_input = st.text_input("Starting URL", value=DEFAULT_CRAWL_URL, help="The URL to begin crawling from (e.g., https://seths.blog/).")
    
    col1, col2 = st.columns(2)
    with col1:
        max_pages_input = st.number_input("Max Pages to Crawl", min_value=1, value=DEFAULT_MAX_PAGES, help="Maximum number of pages to collect.")
    with col2:
        max_depth_input = st.number_input("Max Crawl Depth", min_value=1, value=DEFAULT_MAX_DEPTH, help="Maximum depth for link traversal from the start URL.")
    
    crawl_delay_input = st.number_input("Crawl Delay (ms)", min_value=0, value=DEFAULT_CRAWL_DELAY_MS, help="Delay between HTTP requests to be polite (milliseconds).")

    crawl_submitted = st.form_submit_button("Start Crawling & Indexing")

    if crawl_submitted:
        if not start_url_input:
            st.error("Please enter a valid starting URL.")
        else:
            try:
                parsed_url = urlparse(start_url_input)
                if not parsed_url.scheme or not parsed_url.netloc:
                    st.error("Invalid URL format. Please include scheme (e.g., http:// or https://).")
                else:
                    st.session_state.crawl_status = {"message": f"Crawling '{start_url_input}'...", "success": None}
                    st.empty() # Clear previous messages
                    with st.spinner(f"Crawling and indexing {start_url_input}... This might take a while."):
                        crawl_payload = {
                            "start_url": start_url_input,
                            "max_pages": max_pages_input,
                            "max_depth": max_depth_input,
                            "crawl_delay_ms": crawl_delay_input
                        }
                        response = requests.post(CRAWL_ENDPOINT, json=crawl_payload, timeout=600) # Long timeout for crawl
                        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                        crawl_result = response.json()

                        st.session_state.crawled_urls = crawl_result.get("urls", [])
                        st.session_state.vector_count = crawl_result.get("vector_count", 0)

                        if crawl_result.get("page_count", 0) > 0:
                            st.session_state.crawl_status = {
                                "message": f"Successfully crawled {crawl_result['page_count']} pages and indexed {crawl_result['vector_count']} vectors.",
                                "success": True
                            }
                            st.success(st.session_state.crawl_status["message"])
                            if crawl_result.get("skipped_count", 0) > 0:
                                st.warning(f"Skipped {crawl_result['skipped_count']} URLs during crawl.")
                            if crawl_result.get("errors"):
                                st.error(f"Errors during indexing: {crawl_result['errors']}")
                        else:
                            st.session_state.crawl_status = {
                                "message": f"Crawled 0 pages for {start_url_input}. Check URL or service logs.",
                                "success": False
                            }
                            st.warning(st.session_state.crawl_status["message"])

            except requests.exceptions.RequestException as e:
                error_message = f"API Error during crawl: {e}"
                if hasattr(e, 'response') and e.response is not None:
                    error_message += f"\nResponse: {e.response.text}"
                st.session_state.crawl_status = {"message": error_message, "success": False}
                st.error(error_message)
            except json.JSONDecodeError:
                error_message = f"Failed to decode JSON response from API. Response: {response.text}"
                st.session_state.crawl_status = {"message": error_message, "success": False}
                st.error(error_message)
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                st.session_state.crawl_status = {"message": error_message, "success": False}
                st.error(error_message)
            
            st.rerun() # Rerun to update sidebar metrics immediately


# 2. Ask Question Section
st.header("❓ Ask a Question")
st.write("Query the content that has been crawled and indexed.")

if st.session_state.vector_count == 0:
    st.warning("No content has been indexed yet. Please crawl a website first.")
else:
    with st.form("ask_form"):
        question_input = st.text_area("Your Question", value=st.session_state.last_question, placeholder="E.g., What is Seth Godin's philosophy on 'the dip'?", height=100)
        top_k_input = st.number_input("Number of Chunks to Retrieve (top_k)", min_value=1, value=DEFAULT_TOP_K, help="How many most relevant chunks to retrieve from the vector store.")
        
        ask_submitted = st.form_submit_button("Get Answer")

        if ask_submitted:
            if not question_input:
                st.error("Please enter a question.")
            else:
                st.session_state.last_question = question_input # Store for next run
                st.session_state.last_answer = None # Clear previous answer
                
                with st.spinner("Getting answer..."):
                    try:
                        ask_payload = {
                            "question": question_input,
                            "top_k": top_k_input
                        }
                        response = requests.post(ASK_ENDPOINT, json=ask_payload, timeout=120) # Long timeout for LLM
                        response.raise_for_status()
                        ask_result = response.json()

                        st.session_state.last_answer = ask_result
                        st.success("Answer Retrieved!")

                    except requests.exceptions.RequestException as e:
                        error_message = f"API Error during ask: {e}"
                        if hasattr(e, 'response') and e.response is not None:
                            error_message += f"\nResponse: {e.response.text}"
                        st.error(error_message)
                    except json.JSONDecodeError:
                        error_message = f"Failed to decode JSON response from API. Response: {response.text}"
                        st.error(error_message)
                    except Exception as e:
                        error_message = f"An unexpected error occurred: {e}"
                        st.error(error_message)
                st.rerun() # Rerun to display answer immediately


if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("💡 Answer")
    st.write(st.session_state.last_answer.get("answer", "No answer provided."))

    st.subheader("🔗 Sources")
    sources = st.session_state.last_answer.get("sources", [])
    if sources:
        # Create a dictionary to group snippets by URL to show unique URLs
        grouped_sources = {}
        for source in sources:
            url = source.get("url", "N/A")
            snippet = source.get("snippet", "No snippet available.")
            if url not in grouped_sources:
                grouped_sources[url] = []
            grouped_sources[url].append(snippet)
        
        for url, snippets in grouped_sources.items():
            st.markdown(f"- **URL:** [{url}]({url})")
            with st.expander(f"Show {len(snippets)} snippets from this source"):
                for i, snippet in enumerate(snippets):
                    st.markdown(f"**Snippet {i+1}:**")
                    st.code(snippet, language="text")
                    # st.markdown(f"_{snippet}_") # Use markdown if you prefer formatting over a code block
    else:
        st.info("No specific sources were cited for this answer (e.g., if it was a refusal).")

    st.subheader("⏱️ Timings")
    timings = st.session_state.last_answer.get("timings", {})
    if timings:
        st.json(timings)
    else:
        st.info("No timing information available.")