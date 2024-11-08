import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

# Constants
ROOT_FOLDER = "restaurants"
FILENAME = "restaurants.txt"

def commit_html(folder_number: int, restaurant_name: str, url: str) -> int:
    """
    Downloads and saves the HTML content of a Michelin restaurant page to a specified directory.
    
    Args:
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

        # Fetch HTML content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses

        # Parse HTML to check for access errors
        soup = BeautifulSoup(response.content, 'html.parser')
        if "403 ERROR" in soup.find("h1").text.strip():
            raise PermissionError(f"403 FORBIDDEN: Access denied for {url}")

        # Write the HTML content to the specified file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"'{url}' is saved to '{file_path}'")        
        return 0

    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
        return 1
    except Exception as e:
        print(f"Error for {url}: {e}")
        return 1
    

def crawl_restaurants(base_filename: str = FILENAME, n_pages: int = 100, n_cards: int = 20):
    """
    Crawls Michelin restaurant URLs from a text file and saves each page's HTML locally.

    Args:
        base_filename (str): Path to the text file containing restaurant URLs.
        n_pages (int): Number of pages (groups of restaurants) to process.
        n_cards (int): Number of restaurant URLs to process per page.

    Reads URLs from `base_filename` and divides them into pages, processing `n_cards` URLs per page.
    Repeats each page's download if any of the URLs fails, ensuring all URLs on a page are saved
    successfully before proceeding to the next page.
    """
    try:
        with open(base_filename, 'r') as urls_file:
            for page in range(1, n_pages + 1):
                print(f"CURRENT PAGE: {page}")
                
                # Read a page of URLs, or stop if we reach the end of the file
                page_urls = [urls_file.readline().strip() for _ in range(n_cards)]
                # Remove empty lines if at end of file
                page_urls = [url for url in page_urls if url] 
                
                if not page_urls:
                    print("End of file reached.")
                    break
                
                all_saved = False
                while not all_saved:
                    statuses = [
                        commit_html(page, os.path.basename(url), f"https://guide.michelin.com{url}")
                        for url in page_urls
                    ]
                    # Only move to next page if all files were saved successfully
                    all_saved = sum(statuses) == 0

    except FileNotFoundError:
        print(f"Error: The file '{base_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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
