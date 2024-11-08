import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

# Constants
ROOT_FOLDER = "restaurants"
FILENAME = "restaurants.txt"
# Number of URLs per folder
N_CARDS_PER_FOLDER = 20 
# Maximum retry attempts for each folder 
MAX_FOLDER_RETRIES = 3  

def scraping(filename=FILENAME, n_pages=100):
    """
    Crawls restaurant links from Michelin's website and saves them to a text file.
    Args:
        filename (str): Name of the file to save the restaurant URLs.
        n_pages (int): Number of pages to scrape.
    This function scrapes restaurant listing pages and extracts links for individual
    restaurant pages. Each URL is written to a new line in `filename`.
    """
    link_counter = 1
    with open(filename, 'a') as f:
        for page_num in tqdm(range(1, n_pages + 1)):
            page_url = f"https://guide.michelin.com/en/it/restaurants/page/{page_num}"
            print(f"Processing: {page_url}")
            try:
                # Fetch and parse the listing page
                response = requests.get(page_url)
                # Raise an error for HTTP issues
                response.raise_for_status()  
                list_soup = BeautifulSoup(response.text, 'html.parser')
                # Find restaurant links
                for card in list_soup.find_all("div", {"class": "card__menu"}):
                    card_link = card.find("a")
                    if card_link and card_link.get("href"):
                        card_url = card_link["href"]
                        print(f"{link_counter}: {card_url}")
                        f.write(card_url + "\n")
                        link_counter += 1
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch {page_url}: {e}")
                continue  # Skip to the next page if there's a fetch error

async def commit_html(session, folder_number, restaurant_name, url) -> int:
    """
    Asynchronously downloads and saves the HTML content of a Michelin restaurant page.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session for managing connections.
        folder_number (int): The folder number to categorize saved HTML files.
        restaurant_name (str): The name of the restaurant, used as the filename for the HTML.
        url (str): The full URL of the Michelin restaurant page.
    
    Returns:
        int: 0 if the HTML was downloaded and saved successfully, 1 if any exception occurred.
    """
    try:
        # Create a directory for the specified page if it doesn't exist
        directory = os.path.join(os.getcwd(), ROOT_FOLDER, str(folder_number))
        os.makedirs(directory, exist_ok=True)

        # File path for saving HTML content
        safe_restaurant_name = os.path.basename(restaurant_name)
        file_path = os.path.join(directory, f"{safe_restaurant_name}.html")

        # Fetch HTML content asynchronously
        async with session.get(url) as response:
            if response.status == 403:
                print(f"403 Forbidden: Access denied for {url}")
                return 1

            content = await response.read()

            # Write the HTML content to the specified file
            with open(file_path, 'wb') as file:
                file.write(content)
            print(f"'{url}' is saved to '{file_path}'")
        return 0  # Success

    except Exception as e:
        print(f"Error for {url}: {e}")
        return 1  # Failure


async def process_folder(session, page, urls):
    """
    Processes a single folder of URLs (20 URLs) asynchronously, with retry for failed folders.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests.
        page (int): The page number corresponding to the folder.
        urls (list): List of URLs to be processed in this folder.
    
    Returns:
        int: 0 if all URLs in the folder were downloaded successfully, 1 if any URL failed after retries.
    """
    for attempt in range(MAX_FOLDER_RETRIES):
        print(f"Processing folder {page}, attempt {attempt + 1}/{MAX_FOLDER_RETRIES}")
        
        tasks = [
            commit_html(session, page, os.path.basename(url), f"https://guide.michelin.com{url}")
            for url in urls
        ]

        # Run all tasks concurrently and collect results
        results = await asyncio.gather(*tasks)

        # Check if all URLs in the folder were successful
        if all(result == 0 for result in results):
            print(f"Folder {page} downloaded successfully.")
            return 0  # Success, no need to retry further
        else:
            print(f"Some restaurants in page {page} are failed to be downloaded. Retrying...")

    print(f"Folder {page} failed after {MAX_FOLDER_RETRIES} attempts.")
    return 1


async def crawl_restaurants(base_filename: str = FILENAME, n_cards: int = N_CARDS_PER_FOLDER):
    """
    Asynchronously crawls Michelin restaurant URLs from a text file and saves each page's HTML locally,
    with retries for any failed folders.

    Args:
        base_filename (str): Path to the text file containing restaurant URLs.
        n_cards (int): Number of restaurant URLs to process per page.
    """
    try:
        with open(base_filename, 'r') as urls_file:
            urls = [url.strip() for url in urls_file if url.strip()]

        # Calculate the actual total number of folders required
        total_folders = (len(urls) + n_cards - 1) // n_cards  # Round up to cover all URLs


        async with aiohttp.ClientSession() as session:
            # Process each "folder" (20 URLs) concurrently as separate tasks
            for page in range(1, total_folders + 1):
                print(f"CURRENT PAGE: {page}")

                # Get a subset of URLs for this page (up to 20 URLs per folder)
                start_index = (page - 1) * n_cards
                # Handle end of the file index
                end_index = min(page * n_cards, len(urls))
                page_urls = urls[start_index:end_index]

                # Process this folder (group of 20 URLs) with retries
                result = await process_folder(session, page, page_urls)
                if result == 0:
                    print(f"Folder {page} completed successfully.")
                else:
                    print(f"Folder {page} had failures after maximum retries.")

    except FileNotFoundError:
        print(f"Error: The file '{base_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

