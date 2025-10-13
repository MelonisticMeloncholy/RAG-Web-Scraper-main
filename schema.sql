-- pages table to store crawled URLs and their raw content
CREATE TABLE IF NOT EXISTS pages (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    base_domain TEXT NOT NULL,
    content TEXT NOT NULL,
    crawled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- chunks table to store text chunks derived from pages
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    text_content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding_id INTEGER, -- Will be used to map to FAISS index
    UNIQUE (page_id, chunk_index)
);