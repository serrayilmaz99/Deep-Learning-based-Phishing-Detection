import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
import random
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from pymongo import MongoClient
import re

logging.basicConfig(
    filename='scraping.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def is_website_accessible(url):
    try:
        response = requests.get(url, timeout=10, allow_redirects=True, verify=True)
        print(f"URL: {url}, Status Code: {response.status_code}")
        if response.status_code == 200:
            logging.info(f"Accessible: {url}, Status Code: {response.status_code}")
            return True
        else:
            logging.warning(f"Inaccessible ({response.status_code}): {url}")
            return False
    except requests.exceptions.SSLError:
        logging.error(f"SSL Error: {url}")
        return False
    except requests.RequestException as e:
        logging.error(f"Request Exception for {url}: {e}")
        return False

def setup_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}") 

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)  
    return driver

def scrape_websites(urls, label, db_collection):
    driver = setup_selenium_driver()
    for url in urls:
        if db_collection.find_one({"url": url}):
            logging.info(f"URL already exists in database: {url}")
            continue

        document = {'url': url, 'html_content': None, 'label': label}

        if is_website_accessible(url):
            try:
                start_time = time.time()
                driver.get(url)
                document['html_content'] = driver.page_source
                elapsed_time = time.time() - start_time
                logging.info(f"Successfully scraped: {url} in {elapsed_time:.2f} seconds")
            except TimeoutException:
                logging.warning(f"Timeout while trying to load {url}. Skipping.")
                continue
            except WebDriverException as e:
                logging.error(f"Selenium error scraping {url}: {e}")
                continue
        else:
            logging.info(f"Skipped scraping: {url}")

        if document['html_content']:
            db_collection.insert_one(document)
            logging.info(f"Data inserted into MongoDB for: {url}")
        else:
            logging.info(f"Content is null, skipping insertion for: {url}")

        time.sleep(random.uniform(1, 3)) 
    driver.quit()

def read_urls_from_file(file_path, ty):
    if ty == 'b':
        with open(file_path, 'r') as file:
            urls = []
            for line in file:
                match = re.search(r'\d+,(.+)', line)
                if match:
                    url = match.group(1).strip()
                    if not url.startswith(("http://", "https://")):
                        url = f"http://{url}" 
                    urls.append(url)
            return urls
    else:
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
            return [url if url.startswith(("http://", "https://")) else f"http://{url}" for url in urls]

def connect_to_mongodb(db_name, collection_name):
    client = MongoClient('####', ssl=True)
    db = client[db_name]
    return db[collection_name]

if __name__ == "__main__":
    benign_urls = read_urls_from_file('benign.txt', 'b')
    phishing_urls = read_urls_from_file('phishing.txt', 'p')

    collection = connect_to_mongodb(db_name='phishing', collection_name='phishing')

    scrape_websites(benign_urls, 'benign', collection)

    scrape_websites(phishing_urls, 'phishing', collection)

    logging.info("Scraping completed and data saved to MongoDB.")
