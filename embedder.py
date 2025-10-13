from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Embedder:
    def __init__(self):
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model {model_name}: {e}")
            raise

    def embed_chunks(self, texts):
        """
        Generates embeddings for a list of text chunks.
        """
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

# Example Usage (for testing)
if __name__ == "__main__":
    embedder = Embedder()
    texts = [
        "This is a test sentence for embedding.",
        "Another sentence to demonstrate embedding.",
        "Machine learning is a fascinating field."
    ]
    embeddings = embedder.embed_chunks(texts)
    print(f"Generated {len(embeddings)} embeddings, each with shape: {embeddings[0].shape}")
    # print(embeddings)