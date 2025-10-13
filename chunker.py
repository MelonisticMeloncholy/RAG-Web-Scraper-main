import re
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextChunker:
    def __init__(self, chunk_size=700, chunk_overlap=70):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        if self.chunk_overlap >= self.chunk_size:
            logging.warning(f"Chunk overlap ({self.chunk_overlap}) is greater than or equal to chunk size ({self.chunk_size}). Adjusting overlap to be half of chunk size.")
            self.chunk_overlap = self.chunk_size // 2

    def _split_by_sentences(self, text):
        # A more robust sentence splitting could use NLTK, but regex is simpler for this project scope.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text):
        """
        Chunks text into smaller pieces with a specified overlap.
        Prioritizes sentence boundaries if possible, falling back to character-based.
        """
        chunks = []
        sentences = self._split_by_sentences(text)
        current_chunk_sentences = []
        current_chunk_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            # If adding the next sentence exceeds chunk size, or if it's the very first sentence,
            # create a new chunk.
            if current_chunk_len + sentence_len > self.chunk_size and current_chunk_sentences:
                full_chunk_text = " ".join(current_chunk_sentences)
                chunks.append(full_chunk_text)

                # Start new chunk with overlap
                overlap_text = full_chunk_text[-self.chunk_overlap:].strip()
                # Find a good split point for overlap, e.g., last full sentence in overlap
                overlap_sentences = self._split_by_sentences(overlap_text)
                if overlap_sentences:
                    current_chunk_sentences = overlap_sentences[-1:] # Start with last sentence of overlap
                    current_chunk_len = len(current_chunk_sentences[0])
                else: # If overlap is too small to form a sentence, just use characters
                    current_chunk_sentences = [overlap_text]
                    current_chunk_len = len(overlap_text)
                current_chunk_sentences.append(sentence)
                current_chunk_len += sentence_len
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_len += sentence_len

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # Fallback for very long chunks if sentence splitting wasn't effective
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5: # If a chunk is still excessively long
                for i in range(0, len(chunk), self.chunk_size - self.chunk_overlap):
                    final_chunks.append(chunk[i:i + self.chunk_size])
            else:
                final_chunks.append(chunk)

        # Remove empty chunks and strip whitespace
        return [c.strip() for c in final_chunks if c.strip()]

# Example Usage (for testing)
if __name__ == "__main__":
    chunker = TextChunker()
    long_text = "This is the first sentence. This is the second sentence. This is a very long third sentence that goes on and on and on and on, filling up much space. It has many words. This is the fourth sentence. Finally, the fifth sentence concludes the paragraph." * 5
    chunks = chunker.chunk_text(long_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: (Length: {len(chunk)})\n{chunk}\n---")