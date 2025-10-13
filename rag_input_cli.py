import requests
import json
import time
import threading
import uvicorn
import os
from dotenv import load_dotenv
import logging
from http import HTTPStatus
from urllib.parse import urlparse
import argparse # Import argparse for command-line arguments

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Import your FastAPI app and RAGPipeline for cleanup ---
# Assuming your FastAPI app instance is named 'app' in main.py
# and RAGPipeline is a class in rag_pipeline.py within the same package.
# You might need to adjust these imports based on your exact project structure.
try:
    from main import app # Import your FastAPI app instance
    from rag_pipeline import RAGPipeline # Import the RAGPipeline class for direct cleanup
    from vector_store import VectorStore # Import VectorStore for explicit reset
except ImportError as e:
    logger.error(f"Failed to import app, RAGPipeline, or VectorStore. Ensure main.py, rag_pipeline.py, and vector_store.py are accessible: {e}")
    logger.error("Please check your current directory and PYTHONPATH.")
    exit(1)

# --- Configuration for the test server ---
TEST_SERVER_HOST = "127.0.0.1"
TEST_SERVER_PORT = 8001 # Use a different port to avoid conflicts
BASE_URL = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 30 # seconds

# --- Test Data (will be overridden by command-line arguments) ---
DEFAULT_CRAWL_URL = "https://seths.blog/"
DEFAULT_ANSWERABLE_QUESTION = "What is a common theme or advice Seth Godin gives about marketing?"
DEFAULT_UNANSWERABLE_QUESTION = "What is the square root of pi?"

TEST_MAX_PAGES = 5
TEST_MAX_DEPTH = 1
TEST_CRAWL_DELAY_MS = 200

class TestRunner:
    def __init__(self):
        self.server_thread = None
        self.rag_pipeline_instance = RAGPipeline() # For direct database cleanup
        
        # Initialize with default values, will be set by run_all_tests
        self.crawl_url = DEFAULT_CRAWL_URL
        self.answerable_q = DEFAULT_ANSWERABLE_QUESTION
        self.unanswerable_q = DEFAULT_UNANSWERABLE_QUESTION

    def _run_server(self):
        """Runs the FastAPI application using Uvicorn."""
        logger.info(f"Starting Uvicorn server on {TEST_SERVER_HOST}:{TEST_SERVER_PORT}...")
        uvicorn.run(app, host=TEST_SERVER_HOST, port=TEST_SERVER_PORT, log_level="warning")

    def start_server(self):
        """Starts the FastAPI server in a separate thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for the server to become available
        logger.info(f"Waiting for server to start at {BASE_URL}...")
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{BASE_URL}/docs", timeout=1) # Hit an endpoint to check
                if response.status_code == HTTPStatus.OK:
                    logger.info("Server started successfully!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                logger.warning(f"Unexpected error checking server status: {e}")
            time.sleep(0.5) # Wait a bit before retrying
        
        logger.error(f"Server did not start within {SERVER_STARTUP_TIMEOUT} seconds.")
        return False

    def stop_server(self):
        """Stops the FastAPI server (best effort for daemon threads)."""
        logger.info("Attempting to stop server...")
        # Daemon threads typically stop when the main program exits.
        # For a cleaner shutdown, you'd usually pass a stop event or use uvicorn's Server class directly.
        # For this simple test, letting the daemon thread die is acceptable.
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Server thread is still alive, will terminate with main script.")
        logger.info("Server shutdown initiated.")


    def _clear_database(self):
        """Clears all data from pages and chunks tables directly."""
        logger.info("Clearing existing database data...")
        try:
            self.rag_pipeline_instance.db_manager.clear_all_data()
            # Also clear the in-memory vector store in case it was loaded from disk
            # We need to re-initialize it to ensure it's empty and saved.
            if self.rag_pipeline_instance.embedder.model: # Check if embedder model is loaded
                embedding_dim = self.rag_pipeline_instance.embedder.model.get_sentence_embedding_dimension()
                self.rag_pipeline_instance.vector_store = VectorStore(embedding_dim=embedding_dim)
            else:
                # If embedder model somehow isn't loaded yet, fall back to a default dim
                logger.warning("Embedder model not loaded, using default embedding_dim for vector store reset.")
                self.rag_pipeline_instance.vector_store = VectorStore() # Uses default dim
            
            # Ensure the vector store is saved clean if it was initialized
            self.rag_pipeline_instance.vector_store.save()
            logger.info("Database and vector store successfully cleared.")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise

    def test_crawl_and_index(self):
        """Tests the /crawl endpoint."""
        logger.info(f"\n--- Running Crawl and Index Test for {self.crawl_url} ---")
        endpoint = f"{BASE_URL}/crawl"
        payload = {
            "start_url": self.crawl_url,
            "max_pages": TEST_MAX_PAGES,
            "max_depth": TEST_MAX_DEPTH,
            "crawl_delay_ms": TEST_CRAWL_DELAY_MS
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=600) # Increased timeout for crawling
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            assert response.status_code == HTTPStatus.OK, f"Expected 200, got {response.status_code}"
            assert isinstance(result, dict), "Response is not a dictionary"
            assert "page_count" in result, "page_count missing from response"
            assert "skipped_count" in result, "skipped_count missing from response"
            assert "urls" in result, "urls missing from response"
            assert "vector_count" in result, "vector_count missing from response"

            # Check for reasonable values
            assert result["page_count"] > 0, "No pages were crawled."
            assert result["vector_count"] > 0, "No vectors were indexed."
            assert len(result["urls"]) == result["page_count"], "Number of URLs doesn't match page_count"

            logger.info(f"Crawl and Index Test PASSED.")
            logger.info(f"Crawled {result['page_count']} pages, indexed {result['vector_count']} vectors.")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Crawl and Index Test FAILED: Network or HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
        except AssertionError as e:
            logger.error(f"Crawl and Index Test FAILED: Assertion error: {e}")
        except json.JSONDecodeError:
            logger.error(f"Crawl and Index Test FAILED: Invalid JSON response: {response.text}")
        except Exception as e:
            logger.error(f"Crawl and Index Test FAILED: An unexpected error occurred: {e}")
        return False

    def test_ask_answerable(self):
        """Tests the /ask endpoint with an answerable question."""
        logger.info(f"\n--- Running Ask Test (Answerable) ---")
        endpoint = f"{BASE_URL}/ask"
        payload = {
            "question": self.answerable_q,
            "top_k": int(os.getenv("RETRIEVAL_TOP_K", 5))
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=120) # LLM generation can be slow
            response.raise_for_status()
            result = response.json()

            assert response.status_code == HTTPStatus.OK, f"Expected 200, got {response.status_code}"
            assert "answer" in result, "answer missing from response"
            assert "sources" in result, "sources missing from response"
            assert "timings" in result, "timings missing from response"

            # Check if an answer is provided and not a refusal
            assert result["answer"].strip() != "Not enough information found in crawled content to answer your question.", "Answer was a refusal for an answerable question."
            assert len(result["answer"]) > 50, "Answer is too short or empty." # Arbitrary length check

            # Check for sources
            assert len(result["sources"]) > 0, "No sources provided for an answerable question."
            for source in result["sources"]:
                assert "url" in source, "Source missing URL"
                assert "snippet" in source, "Source missing snippet"
                assert urlparse(source["url"]).netloc != '', f"Invalid URL in source: {source['url']}"

            logger.info(f"Ask Test (Answerable) PASSED.")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Sources: {[s['url'] for s in result['sources']]}")
            logger.info(f"Timings: {result['timings']}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Ask Test (Answerable) FAILED: Network or HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
        except AssertionError as e:
            logger.error(f"Ask Test (Answerable) FAILED: Assertion error: {e}")
        except json.JSONDecodeError:
            logger.error(f"Ask Test (Answerable) FAILED: Invalid JSON response: {response.text}")
        except Exception as e:
            logger.error(f"Ask Test (Answerable) FAILED: An unexpected error occurred: {e}")
        return False

    def test_ask_unanswerable(self):
        """Tests the /ask endpoint with an unanswerable question, expecting a refusal."""
        logger.info(f"\n--- Running Ask Test (Unanswerable) ---")
        endpoint = f"{BASE_URL}/ask"
        payload = {
            "question": self.unanswerable_q,
            "top_k": int(os.getenv("RETRIEVAL_TOP_K", 5))
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            assert response.status_code == HTTPStatus.OK, f"Expected 200, got {response.status_code}"
            assert "answer" in result, "answer missing from response"
            assert "sources" in result, "sources missing from response"
            assert "timings" in result, "timings missing from response"

            # Check if the answer is a refusal
            assert result["answer"].strip() == "Not enough information found in crawled content to answer your question.", "Answer was not a refusal for an unanswerable question."
            # Sources should ideally be empty or only contain unhelpful snippets for refusal
            # assert len(result["sources"]) == 0, "Sources provided for an unanswerable question."

            logger.info(f"Ask Test (Unanswerable) PASSED.")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Sources: {result['sources']}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Ask Test (Unanswerable) FAILED: Network or HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
        except AssertionError as e:
            logger.error(f"Ask Test (Unanswerable) FAILED: Assertion error: {e}")
        except json.JSONDecodeError:
            logger.error(f"Ask Test (Unanswerable) FAILED: Invalid JSON response: {response.text}")
        except Exception as e:
            logger.error(f"Ask Test (Unanswerable) FAILED: An unexpected error occurred: {e}")
        return False

    def run_all_tests(self, crawl_url, answerable_q, unanswerable_q):
        """Orchestrates the entire test suite."""
        self.crawl_url = crawl_url
        self.answerable_q = answerable_q
        self.unanswerable_q = unanswerable_q

        overall_status = True

        try:
            # 1. Clear database before starting (important for clean state)
            self._clear_database()

            # 2. Start the FastAPI server
            if not self.start_server():
                return False # Cannot proceed if server fails to start

            # Give a small buffer time after server confirms start
            time.sleep(2)

            # 3. Run crawl and index test
            if not self.test_crawl_and_index():
                overall_status = False
                logger.error("Crawl and Index test failed, subsequent tests may also fail due to lack of data.")
                # If crawl fails, asking questions will definitely fail for lack of data.
                # We can stop here or let them run to see how they handle empty data.
                # For this script, we'll let them run, as they should ideally refuse.
            else:
                # Only run ask tests if crawl was successful, otherwise it's expected to fail/refuse
                # Give some time for index files to be written to disk if they are
                time.sleep(1) 
                # 4. Run answerable question test
                if not self.test_ask_answerable():
                    overall_status = False

                # 5. Run unanswerable question test
                if not self.test_ask_unanswerable():
                    overall_status = False

        except Exception as e:
            logger.critical(f"A critical error occurred during test execution: {e}")
            overall_status = False
        finally:
            self.stop_server() # Attempt to stop server gracefully

        logger.info(f"\n--- ALL TESTS {'PASSED' if overall_status else 'FAILED'} ---")
        return overall_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end RAG service tests.")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_CRAWL_URL,
        help=f"The starting URL to crawl. Default: {DEFAULT_CRAWL_URL}"
    )
    parser.add_argument(
        "--answerable_q",
        type=str,
        default=DEFAULT_ANSWERABLE_QUESTION,
        help=f"A question expected to be answerable by the crawled content. Default: {DEFAULT_ANSWERABLE_QUESTION}"
    )
    parser.add_argument(
        "--unanswerable_q",
        type=str,
        default=DEFAULT_UNANSWERABLE_QUESTION,
        help=f"A question expected to be unanswerable by the crawled content. Default: {DEFAULT_UNANSWERABLE_QUESTION}"
    )

    args = parser.parse_args()

    test_runner = TestRunner()
    try:
        if not test_runner.run_all_tests(args.url, args.answerable_q, args.unanswerable_q):
            exit(1) # Exit with a non-zero code if tests fail
    finally:
        # Ensure DB connection is closed even if an error occurs early
        test_runner.rag_pipeline_instance.db_manager.close()
    
    logger.info("Test script finished.")