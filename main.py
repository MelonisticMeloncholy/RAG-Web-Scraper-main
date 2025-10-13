from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import logging
import uvicorn
from urllib.parse import urlparse

# Adjust relative import based on your project structure
# If main.py is in the root and rag_pipeline.py is in a subfolder,
# you might need to adjust __init__.py or sys.path
from rag_pipeline import RAGPipeline 

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="RAG Service API",
    description="A small Retrieval-Augmented Generation (RAG) service for website content.",
    version="0.1.0"
)

# Initialize RAGPipeline globally or use dependency injection if preferred
rag_pipeline = RAGPipeline()

class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_pages: Optional[int] = int(os.getenv("CRAWL_MAX_PAGES", 50))
    max_depth: Optional[int] = int(os.getenv("CRAWL_MAX_DEPTH", 3))
    crawl_delay_ms: Optional[int] = int(os.getenv("CRAWL_DELAY_MS", 200))

class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    urls: List[HttpUrl]
    vector_count: Optional[int] = None
    errors: Optional[List[str]] = None

class IndexRequest(BaseModel):
    # In this integrated pipeline, index is part of crawl_and_index
    # but for explicit control, you might define it separately.
    # For this project, crawl_and_index implicitly handles both.
    # If you want to allow re-indexing only, without recrawling, you'd
    # need to fetch content from DB here.
    # For simplicity, we'll keep index tied to crawl for now.
    pass

class IndexResponse(BaseModel):
    vector_count: int
    errors: List[str]

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = int(os.getenv("RETRIEVAL_TOP_K", 5))

class SourceSnippet(BaseModel):
    url: HttpUrl
    snippet: str

class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    timings: Timings

@app.post("/crawl", response_model=CrawlResponse, status_code=status.HTTP_200_OK)
async def crawl_and_index_endpoint(request: CrawlRequest):
    """
    Initiates crawling a website from a starting URL and indexes its content.
    This endpoint performs both crawling and indexing steps.
    """
    logging.info(f"Received crawl request for: {request.start_url}")
    
    # Ensure the domain limitation is respected for the starting URL itself
    parsed_start_url = urlparse(str(request.start_url))
    if not parsed_start_url.netloc:
         raise HTTPException(status_code=400, detail="Invalid start_url: could not parse domain.")

    # For simplicity, if the pipeline is already running an index, it might overwrite.
    # In a real system, you'd queue tasks or manage states.
    try:
        result = rag_pipeline.crawl_and_index(
            start_url=str(request.start_url),
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            crawl_delay_ms=request.crawl_delay_ms
        )
        return CrawlResponse(**result)
    except Exception as e:
        logging.exception(f"Error during crawl and index for {request.start_url}")
        raise HTTPException(status_code=500, detail=f"Failed to crawl and index: {e}")

# No separate /index endpoint is provided since crawl_and_index handles both
# If a separate index from DB functionality is needed, it would be added here.

@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_endpoint(request: AskRequest):
    """
    Asks a question against the indexed content.
    """
    logging.info(f"Received ask request for question: {request.question}")
    try:
        response = rag_pipeline.ask(request.question, request.top_k)
        # Convert HttpUrl strings in sources to HttpUrl Pydantic type
        for source in response['sources']:
            source['url'] = HttpUrl(source['url'])
        return AskResponse(**response)
    except Exception as e:
        logging.exception(f"Error during ask for question: {request.question}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

# To run the API:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload