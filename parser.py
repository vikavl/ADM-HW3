import os
import re
from bs4 import BeautifulSoup

def normalize_website_url(url):
    """
    Normalizes a website URL to ensure it has the correct format.
    
    - Adds 'https://' if the URL starts with 'www' but is missing the scheme.
    - Returns an empty string if the URL is invalid.
    
    Args:
        url (str): The raw URL extracted from the HTML.
        
    Returns:
        str: The normalized URL or an empty string if the URL is invalid.
    """
    if not url:
        return ""
    
    # Regex to check if URL starts with a valid scheme (http or https)
    scheme_regex = re.compile(r'^(http://|https://)')
    if not scheme_regex.match(url):
        # If URL starts with 'www.', add 'https://' to normalize it
        if url.startswith('www.'):
            url = 'https://' + url
        else:
            # If the URL doesn't start with a valid scheme or 'www.', mark it as invalid
            return ""
    
    return url

def get_restaurant_name(soup):
    """
    Extracts the restaurant name from the parsed HTML. If the name is not found,
    returns an empty string to maintain alignment in the output.
    """
    element = soup.find("a", {"data-restaurant-name": True})
    return element.get("data-restaurant-name") if element else ""

def get_address(soup):
    """
    Extracts the full address, splitting into street, city, postal code, and country.
    Returns empty strings if address parts are missing.
    """
    try:
        address = [
            detail.text.strip() for detail in soup.find("div", {"class": "data-sheet__detail-info"}).find_all("div", {"class": "data-sheet__block--text"})
        ][0]
        address_parts = [remove_newline(item.strip()) for item in address.split(",")]
        # Return exactly four components, adding empty strings if needed
        if len(address_parts) == 5:
            return [", ".join(address_parts[:2]), address_parts[2], address_parts[3], address_parts[4]]
        elif len(address_parts) == 4:
            return address_parts
    except AttributeError:
        pass
    return ["", "", "", ""]

def get_price_range(soup):
    """
    Extracts price range and cuisine type from the HTML. Returns empty strings if not found.
    """
    try:
        cuisine = [
            detail.text.strip() for detail in soup.find("div", {"class": "data-sheet__detail-info"}).find_all("div", {"class": "data-sheet__block--text"})
        ][1]
        price_cuisine = [item.strip() for item in cuisine.split("·")]
        if len(price_cuisine) == 2:
            return price_cuisine
        elif len(price_cuisine) == 1:
            return [price_cuisine[0], ""]
    except (AttributeError, IndexError):
        pass
    return ["", ""]

def get_description(soup):
    """
    Extracts the description from the HTML. Returns an empty string if not found.
    """
    try:
        return remove_newline(soup.find("div", {"class": "restaurant-details"}).find("div", {"class": "data-sheet__description"}).text.strip())
    except AttributeError:
        return ""

def get_facilities_services(soup):
    """
    Extracts a list of facilities and services. Returns an empty list if none found.
    """
    try:
        return [item.text.strip() for service_div in soup.find("div", {"class": "restaurant-details"}).find_all("div", {"class": "restaurant-details__services"}) for item in service_div.find_all("li")]
    except AttributeError:
        return []

def get_services(soup):
    """
    Extracts a list of accepted credit card icons. Returns an empty list if none found.
    """
    restaurant_details_div = soup.find("div", {"class": "restaurant-details"})
    if restaurant_details_div:
        services_info_div = restaurant_details_div.find("div", {"class": "restaurant-details__services--info"})
        if services_info_div:
            list_card_div = services_info_div.find("div", {"class": "list--card"})
            if list_card_div:
                img_tags = list_card_div.find_all("img")
                card_names = [img.get("data-src").split("/")[-1].split("-")[0] for img in img_tags if img.get("data-src")]
                return card_names if card_names else []
    return []

def get_phone_number(soup):
    """
    Extracts the phone number from the HTML. Returns an empty string if not found.
    """
    element = soup.find("span", {"x-ms-format-detection": "none"})
    return element.text.strip() if element else ""

def get_website(soup):
    """
    Extracts and normalizes the restaurant's website URL from the HTML.
    """
    website_element = soup.find("a", {"href": True, "data-event": "CTA_website"})
    raw_url = website_element.get("href") if website_element else None
    return normalize_website_url(raw_url)

def remove_newline(text):
    """
    Removes newline characters from the text (inlines it).
    """
    # Remove newline characters and replace them with a space
    return " ".join(text.split())

def list_to_string(ls):
    """
    Converts a list to a formatted string.
    """
    # Handle the empty list case
    if not ls:
        return "none"
    
    # Join elements
    line = "[" + ", ".join([f"'{el}'" for el in ls]) + "]"    
    return line

def extract_restaurant_data(html_file_path):
    """
    Extracts structured data for a restaurant from an HTML file using BeautifulSoup.
    
    This function opens and parses an HTML file containing details about a restaurant,
    then extracts specific data points (e.g., name, address, price range) by looking for 
    HTML elements and attributes associated with each field. If a 403 Forbidden error is 
    detected in the HTML, it returns None to skip processing for that file.
    
    Args:
        html_file_path (str): The path to the HTML file to parse.

    Returns:
        dict or None: A dictionary containing the restaurant's data if parsing is successful,
                      or None if the file is inaccessible or data extraction fails.
                      The dictionary includes the following keys:
                        - "restaurantName" (str): The name of the restaurant.
                        - "address" (str): The street address.
                        - "city" (str): The city where the restaurant is located.
                        - "postalCode" (str): The postal code.
                        - "country" (str): The country where the restaurant is located.
                        - "priceRange" (str): The restaurant's price range.
                        - "cuisineType" (str): The type of cuisine served.
                        - "description" (str): A description of the restaurant.
                        - "facilitiesServices" (list of str): A list of available facilities/services.
                        - "creditCards" (list of str): A list of accepted credit card types.
                        - "phoneNumber" (str): The restaurant's contact phone number.
                        - "website" (str): The restaurant's website URL.
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Case: file was not crawled correctly 
    if soup.find("h1") and "403 ERROR" in soup.find("h1").text.strip():
        return None

    restaurant_data = {
        "restaurantName": get_restaurant_name(soup),
        "address": "",
        "city": "",
        "postalCode": "",
        "country": "",
        "priceRange": "",
        "cuisineType": "",
        "description": get_description(soup),
        "facilitiesServices": get_facilities_services(soup),
        "creditCards": get_services(soup),
        "phoneNumber": get_phone_number(soup),
        "website": get_website(soup)
    }

    address, city, postal_code, country = get_address(soup)
    restaurant_data["address"], restaurant_data["city"], restaurant_data["postalCode"], restaurant_data["country"] = address, city, postal_code, country

    price_range, cuisine_type = get_price_range(soup)
    restaurant_data["priceRange"], restaurant_data["cuisineType"] = price_range, cuisine_type

    return restaurant_data

def write_restaurant_data_to_tsv(output_file_path, file_paths):
    """
    Writes extracted restaurant data to a TSV file, handling missing values by converting them to empty strings.
    
    Args:
        output_file_path (str): The path to the output TSV file.
        file_paths (list of str): List of file paths to HTML files to process.
        
    The function writes the data in a tab-separated format with headers at the top. Each row corresponds
    to a restaurant, with empty values where data is missing to ensure consistent column alignment. The
    function appends data to the file if it already exists, adding headers only if the file is new or empty.
    
    Functionality:
        - For each HTML file in file_paths, extracts restaurant data using `extract_restaurant_data`.
        - Writes a row to the TSV file for each restaurant, with missing fields represented as empty strings.
        - Skips entries if a 403 error is detected in the HTML file.
    """
    headers = [
        "restaurantName",
        "address",
        "city",
        "postalCode",
        "country",
        "priceRange",
        "cuisineType",
        "description",
        "facilitiesServices",
        "creditCards",
        "phoneNumber",
        "website"
    ]
    
    if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
        with open(output_file_path, 'w', encoding='utf-8') as f_tsv:
            f_tsv.write('\t'.join(headers) + '\n')

    # Track processed files to avoid duplicates
    processed_files = set()
    skipped_files = set()
    
    with open(output_file_path, 'a', encoding='utf-8') as f_tsv:
        for file_path in file_paths:
            # Skip if file has already been processed
            if file_path in processed_files:
                continue

            # Process and extract data
            restaurant_data = extract_restaurant_data(file_path)
            if restaurant_data is None:
                print(f"{file_path} was not parsed correctly or returned a 403 error.")
                continue

            # Mark file as processed
            processed_files.add(file_path)

            # SEt "none" placeholder
            row_data = [
                restaurant_data.get("restaurantName", "none"),
                restaurant_data.get("address", "none"),
                restaurant_data.get("city", "none"),
                restaurant_data.get("postalCode", "none"),
                restaurant_data.get("country", "none"),
                restaurant_data.get("priceRange", "none"),
                restaurant_data.get("cuisineType", "none"),
                restaurant_data.get("description", "none"),
                list_to_string(restaurant_data.get("facilitiesServices", [])),
                list_to_string(restaurant_data.get("creditCards", [])),
                restaurant_data.get("phoneNumber", "none"),
                restaurant_data.get("website", "none")
            ]

            # Check if all values in row_data are undefined
            if all(value in ["none", None, ""] for value in row_data):
                skipped_files.add(file_path)
                print(f"All values are undefined, skipping this row: {file_path}")
            else:
                tsv_row = '\t'.join(row_data)
                f_tsv.write(tsv_row + '\n')

    if len(skipped_files) == 0: 
        print(f"Reprocessed files: {set.intersection(processed_files, skipped_files)}")
