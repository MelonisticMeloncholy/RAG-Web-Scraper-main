import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from robotexclusionrulesparser import RobotExclusionRulesParser
import time
from dotenv import load_dotenv
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class PoliteCrawler:
    def __init__(self, start_url, max_pages=50, max_depth=3, crawl_delay_ms=200):
        self.start_url = start_url
        self.base_domain = self._get_base_domain(start_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawl_delay = crawl_delay_ms / 1000.0 # Convert ms to seconds
        self.visited_urls = set()
        self.queue = [(start_url, 0)] # (url, depth)
        self.robots_parsers = {} # domain: RobotExclusionRulesParser instance
        self.page_data = [] # List of {'url': ..., 'content': ...}
        self.skipped_urls = []

    def _get_base_domain(self, url):
        return urlparse(url).netloc

    def _get_robots_parser(self, domain):
        if domain not in self.robots_parsers:
            robots_url = f"http://{domain}/robots.txt"
            try:
                response = requests.get(robots_url, timeout=5)
                response.raise_for_status()
                parser = RobotExclusionRulesParser()
                parser.parse(response.text)
                self.robots_parsers[domain] = parser
                logging.info(f"Loaded robots.txt for {domain}")
            except (requests.exceptions.RequestException, ValueError) as e:
                logging.warning(f"Could not fetch or parse robots.txt for {domain}: {e}. Assuming no restrictions.")
                self.robots_parsers[domain] = None # No robots.txt, assume full access
        return self.robots_parsers[domain]

    def _is_allowed(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        if domain != self.base_domain:
            return False # Stay within the registrable domain

        parser = self._get_robots_parser(domain)
        if parser:
            # Use a common user-agent, e.g., 'Googlebot' or a custom one.
            # Project suggests a polite crawler, so let's be explicit.
            return parser.is_allowed('RAGServiceCrawler', url)
        return True # If no robots.txt or error, assume allowed

    def _extract_main_content(self, html_content):
        """
        Extracts main content from HTML, reducing boilerplate.
        This is a heuristic and can be improved.
        """
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove common boilerplate elements
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
            tag.decompose()

        # Try to find a main content area
        main_content_tags = soup.find_all(['article', 'main', 'div', 'section'])
        if main_content_tags:
            # Prioritize larger, central blocks
            best_content = None
            max_len = 0
            for tag in main_content_tags:
                text = tag.get_text(separator=' ', strip=True)
                if len(text) > max_len:
                    max_len = len(text)
                    best_content = text
            if best_content:
                return best_content

        # Fallback to body text if no clear main content found
        if soup.body:
            return soup.body.get_text(separator=' ', strip=True)
        return soup.get_text(separator=' ', strip=True) # Last resort

    def crawl(self):
        while self.queue and len(self.page_data) < self.max_pages:
            current_url, current_depth = self.queue.pop(0)

            if current_url in self.visited_urls or current_depth > self.max_depth:
                continue

            if not self._is_allowed(current_url):
                logging.info(f"Skipping {current_url} due to robots.txt or domain restriction.")
                self.skipped_urls.append(current_url)
                continue

            logging.info(f"Crawling: {current_url} (Depth: {current_depth}, Pages found: {len(self.page_data)})")
            self.visited_urls.add(current_url)

            try:
                time.sleep(self.crawl_delay) # Be polite
                response = requests.get(current_url, timeout=10)
                response.raise_for_status() # Raise an exception for HTTP errors
                html_content = response.text

                # Store the *cleaned* content for indexing
                cleaned_text = self._extract_main_content(html_content)
                self.page_data.append({'url': current_url, 'content': cleaned_text, 'raw_html': html_content})

                # Extract links for further crawling
                if current_depth < self.max_depth:
                    soup = BeautifulSoup(html_content, 'lxml')
                    for link in soup.find_all('a', href=True):
                        absolute_url = urljoin(current_url, link['href'])
                        # Basic validation for HTTP/HTTPS links
                        if absolute_url.startswith(('http://', 'https://')):
                            parsed_link = urlparse(absolute_url)
                            # Only add to queue if it's within the base domain and not yet visited
                            if parsed_link.netloc == self.base_domain and absolute_url not in self.visited_urls:
                                self.queue.append((absolute_url, current_depth + 1))

            except requests.exceptions.RequestException as e:
                logging.error(f"Error crawling {current_url}: {e}")
                self.skipped_urls.append(current_url)
            except Exception as e:
                logging.error(f"An unexpected error occurred for {current_url}: {e}")
                self.skipped_urls.append(current_url)

        logging.info(f"Finished crawling. Total pages collected: {len(self.page_data)}")
        logging.info(f"Total URLs skipped: {len(self.skipped_urls)}")
        return len(self.page_data), len(self.skipped_urls), [d['url'] for d in self.page_data]

# Example Usage (for testing)
if __name__ == "__main__":
    test_url = "https://www.google.com" # Replace with a site you want to test
    crawler = PoliteCrawler(test_url, max_pages=10, max_depth=2, crawl_delay_ms=500)
    page_count, skipped_count, urls = crawler.crawl()
    print(f"Crawled {page_count} pages, skipped {skipped_count} URLs.")
    # for page in crawler.page_data:
    #     print(f"URL: {page['url']}\nContent (snippet): {page['content'][:200]}...\n---")