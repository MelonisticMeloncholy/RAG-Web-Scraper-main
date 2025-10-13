import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DBManager:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set.")
        self.conn = None

    def _get_connection(self):
        if self.conn is None or self.conn.closed != 0:
            try:
                self.conn = psycopg2.connect(self.database_url)
                self.conn.autocommit = True # Auto-commit for simple operations
            except psycopg2.Error as e:
                logging.error(f"Database connection error: {e}")
                raise
        return self.conn

    def insert_page(self, url, base_domain, content):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("INSERT INTO pages (url, base_domain, content) VALUES (%s, %s, %s) ON CONFLICT (url) DO UPDATE SET content = EXCLUDED.content, crawled_at = EXCLUDED.crawled_at RETURNING id"),
                    (url, base_domain, content)
                )
                page_id = cur.fetchone()[0]
                logging.debug(f"Inserted/Updated page {url} with ID {page_id}")
                return page_id
        except psycopg2.Error as e:
            logging.error(f"Error inserting page {url}: {e}")
            raise

    def insert_chunk(self, page_id, text_content, chunk_index, embedding_id=None):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("INSERT INTO chunks (page_id, text_content, chunk_index, embedding_id) VALUES (%s, %s, %s, %s) RETURNING id"),
                    (page_id, text_content, chunk_index, embedding_id)
                )
                chunk_id = cur.fetchone()[0]
                logging.debug(f"Inserted chunk for page {page_id}, index {chunk_index}")
                return chunk_id
        except psycopg2.Error as e:
            logging.error(f"Error inserting chunk for page {page_id}, index {chunk_index}: {e}")
            raise

    def get_chunk_by_id(self, chunk_db_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT c.text_content, p.url FROM chunks c JOIN pages p ON c.page_id = p.id WHERE c.id = %s"),
                    (chunk_db_id,)
                )
                result = cur.fetchone()
                if result:
                    return {'text_content': result[0], 'url': result[1]}
                return None
        except psycopg2.Error as e:
            logging.error(f"Error retrieving chunk {chunk_db_id}: {e}")
            raise

    def get_all_pages(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT id, url, content FROM pages"))
                return cur.fetchall()
        except psycopg2.Error as e:
            logging.error(f"Error fetching all pages: {e}")
            raise

    def get_all_chunks(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT id, page_id, text_content FROM chunks ORDER BY page_id, chunk_index"))
                return cur.fetchall()
        except psycopg2.Error as e:
            logging.error(f"Error fetching all chunks: {e}")
            raise

    def update_chunk_embedding_id(self, chunk_db_id, embedding_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE chunks SET embedding_id = %s WHERE id = %s"),
                    (embedding_id, chunk_db_id)
                )
            logging.debug(f"Updated chunk {chunk_db_id} with embedding_id {embedding_id}")
        except psycopg2.Error as e:
            logging.error(f"Error updating embedding_id for chunk {chunk_db_id}: {e}")
            raise

    def clear_all_data(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE"))
                cur.execute(sql.SQL("TRUNCATE TABLE pages RESTART IDENTITY CASCADE"))
            logging.info("Cleared all data from pages and chunks tables.")
        except psycopg2.Error as e:
            logging.error(f"Error clearing data: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logging.info("Database connection closed.")