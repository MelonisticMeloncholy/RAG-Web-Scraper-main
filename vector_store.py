import faiss
import numpy as np
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, embedding_dim=384): # all-MiniLM-L6-v2 has 384 dimensions
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map = [] # Maps FAISS internal index to our DB chunk ID
        self.next_faiss_id = 0

    def add_embeddings(self, embeddings, chunk_db_ids):
        """
        Adds new embeddings to the FAISS index and stores their corresponding DB IDs.
        """
        if len(embeddings) == 0:
            return
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings).astype('float32')

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")

        if self.index is None:
            # Flat index, suitable for smaller datasets. For larger, consider IndexIVFFlat.
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logging.info(f"Initialized FAISS IndexFlatL2 with dimension {self.embedding_dim}")

        self.index.add(embeddings)
        logging.info(f"Added {len(embeddings)} embeddings to FAISS index.")

        # Update the ID map
        for i, db_id in enumerate(chunk_db_ids):
            self.id_map.append({'faiss_id': self.next_faiss_id + i, 'chunk_db_id': db_id})
        self.next_faiss_id += len(embeddings)


    def search(self, query_embedding, k=5):
        """
        Searches the FAISS index for the top-k nearest neighbors.
        Returns a list of tuples: (distance, chunk_db_id).
        """
        if self.index is None or self.index.ntotal == 0:
            logging.warning("FAISS index is empty. No search performed.")
            return []

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        elif query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, faiss_indices = self.index.search(query_embedding, k)

        results = []
        for dist, faiss_idx in zip(distances[0], faiss_indices[0]):
            if faiss_idx == -1: # FAISS returns -1 for not found
                continue
            # Find the original chunk_db_id using the id_map
            # In a real-world scenario, for very large id_map, this could be slow.
            # A direct mapping (e.g., dictionary) is better if faiss_id is guaranteed to be contiguous/small.
            # For `IndexFlat`, faiss_id maps directly to the insertion order, so id_map[faiss_idx] works.
            try:
                # Given how we add, faiss_idx should correspond to the index in id_map
                chunk_db_id = self.id_map[faiss_idx]['chunk_db_id']
                results.append((dist, chunk_db_id))
            except IndexError:
                logging.error(f"FAISS index {faiss_idx} not found in id_map. This indicates an inconsistency.")
        return results

    def get_total_vectors(self):
        return self.index.ntotal if self.index else 0

    def save(self, faiss_index_path="faiss_index.bin", id_map_path="id_map.pkl"):
        if self.index:
            faiss.write_index(self.index, faiss_index_path)
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            logging.info(f"FAISS index saved to {faiss_index_path} and ID map to {id_map_path}")

    def load(self, faiss_index_path="faiss_index.bin", id_map_path="id_map.pkl"):
        if os.path.exists(faiss_index_path) and os.path.exists(id_map_path):
            self.index = faiss.read_index(faiss_index_path)
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            self.next_faiss_id = len(self.id_map) # Recalculate next ID
            logging.info(f"FAISS index and ID map loaded. {self.get_total_vectors()} vectors loaded.")
            return True
        else:
            logging.warning("FAISS index or ID map files not found. Starting with empty index.")
            return False

# Dummy Embedder class for testing purposes
class Embedder:
    def embed_chunks(self, texts):
        # Returns random embeddings for demonstration
        return np.random.rand(len(texts), 384).astype('float32')

# Example Usage (for testing)
if __name__ == "__main__":
    embedder = Embedder()
    texts = ["apple pie", "orange juice", "fruit salad", "apple phone"]
    db_ids = [101, 102, 103, 104]
    embeddings = embedder.embed_chunks(texts)

    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_embeddings(embeddings, db_ids)

    query = "a sweet apple dessert"
    query_embedding = embedder.embed_chunks([query])[0]
    results = vector_store.search(query_embedding, k=2)
    print(f"\nSearch results for '{query}':")
    for dist, chunk_db_id in results:
        text = texts[db_ids.index(chunk_db_id)] # Simple mapping for example
        print(f"  Distance: {dist:.4f}, DB ID: {chunk_db_id}, Text: '{text}'")

    vector_store.save()
    new_vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    new_vector_store.load()
    print(f"Loaded index has {new_vector_store.get_total_vectors()} vectors.")