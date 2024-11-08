import os
import aiohttp
import asyncio

# Constants
ROOT_FOLDER = "restaurants"
FILENAME = "restaurants.txt"

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

            # Fetch the content without blocking other requests, 
            # pages download in parallel
            content = await response.read()

            # Write the HTML content to the specified file
            with open(file_path, 'wb') as file:
                file.write(content)
            print(f"'{url}' is saved to '{file_path}'")        
        return 0

    except Exception as e:
        print(f"Error for {url}: {e}")
        return 1
    

async def crawl_restaurants(base_filename: str = FILENAME, n_pages: int = 100, n_cards: int = 20):
    """
    Asynchronously crawls Michelin restaurant URLs from a text file and saves each page's HTML locally.

    Args:
        base_filename (str): Path to the text file containing restaurant URLs.
        n_pages (int): Number of pages (groups of restaurants) to process.
        n_cards (int): Number of restaurant URLs to process per page.

    Reads URLs from `base_filename` and divides them into pages, processing `n_cards` URLs per page.
    Repeats each page's download if any of the URLs fails, ensuring all URLs on a page are saved
    successfully before proceeding to the next page.
    """
    # List of commiting tasks
    tasks = []

    try:
        with open(base_filename, 'r') as urls_file:
            urls = [url.strip() for url in urls_file if url.strip()]

        # aiohttp.ClientSession performs asynchronous HTTP requests, 
        # allowing multiple pages to be fetched concurrently
        async with aiohttp.ClientSession() as session:
            for page in range(1, n_pages + 1):
                print(f"CURRENT PAGE: {page}")

                # Get a subset of URLs for this page
                page_urls = urls[(page - 1) * n_cards : page * n_cards]

                if not page_urls:
                    print("End of URLs file is reached.")
                    break

                # Create asynchronous tasks for each URL
                tasks.extend(
                    commit_html(session, page, os.path.basename(url), f"https://guide.michelin.com{url}")
                    for url in page_urls
                )

            # Run all tasks concurrently and wait for completion
            results = await asyncio.gather(*tasks)
            if all(result == 0 for result in results):
                print("All pages downloaded successfully.")
            else:
                print("Some pages failed to download. Check errors above.")

    except FileNotFoundError:
        print(f"Error: The file '{base_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")