from bs4 import BeautifulSoup
import os

def get_restaurant_name(soup):
    element = soup.find("a", {"data-restaurant-name": True})
    return element.get("data-restaurant-name") if element else None

def get_address(soup):
    try:
        address = [
            detail.text.strip() for detail in soup.find("div", {"class": "data-sheet__detail-info"}).find_all("div", {"class": "data-sheet__block--text"})
        ][0]
        address_parts = [item.strip() for item in address.split(",")]
        if len(address_parts) == 5:
            return [", ".join(address_parts[:2]), address_parts[2], address_parts[3], address_parts[4]]
        elif len(address_parts) == 4:
            return address_parts
    except AttributeError:
        return None, None, None, None
    return None, None, None, None

def get_price_range(soup):
    try:
        cuisine = [
            detail.text.strip() for detail in soup.find("div", {"class": "data-sheet__detail-info"}).find_all("div", {"class": "data-sheet__block--text"})
        ][1]
        return [item.strip() for item in cuisine.split("·")]
    except (AttributeError, IndexError):
        return None, None

def get_description(soup):
    try:
        return soup.find("div", {"class": "restaurant-details"}).find("div", {"class": "data-sheet__description"}).text.strip()
    except AttributeError:
        return None

def get_facilities_services(soup):
    try:
        return [item.text.strip() for service_div in soup.find("div", {"class": "restaurant-details"}).find_all("div", {"class": "restaurant-details__services"}) for item in service_div.find_all("li")]
    except AttributeError:
        return None

def get_services(soup):
    restaurant_details_div = soup.find("div", {"class": "restaurant-details"})
    if restaurant_details_div:
        services_info_div = restaurant_details_div.find("div", {"class": "restaurant-details__services--info"})
        if services_info_div:
            list_card_div = services_info_div.find("div", {"class": "list--card"})
            if list_card_div:
                img_tags = list_card_div.find_all("img")
                card_names = [img.get("data-src").split("/")[-1].split("-")[0] for img in img_tags if img.get("data-src")]
                return card_names if card_names else None
    return None

def get_phone_number(soup):
    element = soup.find("span", {"x-ms-format-detection": "none"})
    return element.text.strip() if element else None

def get_website(soup):
    website_element = soup.find("a", {"href": True, "data-event": "CTA_website"})
    return website_element.get("href") if website_element else None

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
                        - "facilitiesServices" (list): A list of available facilities/services.
                        - "creditCards" (list): A list of accepted credit card types.
                        - "phoneNumber" (str): The restaurant's contact phone number.
                        - "website" (str): The restaurant's website URL.
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Check if the page is forbidden
    if soup.find("h1") and "403 ERROR" in soup.find("h1").text.strip():
        return None

    # Initialize restaurant data dictionary for each file to avoid shared state issues
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

    # Address components
    address, city, postal_code, country = get_address(soup)
    restaurant_data["address"], restaurant_data["city"], restaurant_data["postalCode"], restaurant_data["country"] = address, city, postal_code, country

    # Price range and cuisine type
    price_range, cuisine_type = get_price_range(soup)
    restaurant_data["priceRange"], restaurant_data["cuisineType"] = price_range, cuisine_type

    return restaurant_data

def write_restaurant_data_to_tsv(output_file_path):
    """
    Writes structured restaurant data from multiple HTML files into a TSV file.
    
    This function iterates through a directory structure where each subdirectory 
    contains HTML files with restaurant data, extracts information from each file 
    using `extract_restaurant_data`, and writes it to a tab-separated values (TSV) 
    file. Each restaurant's data is saved as a single row in the TSV file, with 
    headers included at the beginning.

    Args:
        output_file_path (str): The path to the TSV file where the data should be saved.
    """
    #  Initialize TSV file with headers
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
    
    # Write the header row only if the file does not exist or is empty
    if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
        with open(output_file_path, 'w', encoding='utf-8') as f_tsv:
            f_tsv.write('\t'.join(headers) + '\n')
    
    # Append restaurant data rows to the TSV file
    with open(output_file_path, 'a', encoding='utf-8') as f_tsv:
        for folder_num in range(1, 101):
            folder_path = os.path.join('restaurants', str(folder_num))
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.html'):
                        file_path = os.path.join(folder_path, filename)
                        print(f"Processing file: {file_path}")
                        restaurant_data = extract_restaurant_data(file_path)
                        if restaurant_data is None:
                            print("403 FORBIDDEN")
                            continue

                        # Build TSV row with explicit handling for None values
                        row_data = [
                            restaurant_data.get("restaurantName") or "",
                            restaurant_data.get("address") or "",
                            restaurant_data.get("city") or "",
                            restaurant_data.get("postalCode") or "",
                            restaurant_data.get("country") or "",
                            restaurant_data.get("priceRange") or "",
                            restaurant_data.get("cuisineType") or "",
                            restaurant_data.get("description") or "",
                            ", ".join(restaurant_data.get("facilitiesServices", [])) if restaurant_data.get("facilitiesServices") else "",
                            ", ".join(restaurant_data.get("creditCards", [])) if restaurant_data.get("creditCards") else "",
                            restaurant_data.get("phoneNumber") or "",
                            restaurant_data.get("website") or ""
                        ]

                        # Verify the correct number of fields in row_data
                        if len(row_data) != len(headers):
                            print(f"Error: Row data length {len(row_data)} does not match header length {len(headers)}")
                            print(f"Row data: {row_data}")
                            continue  # Skip this row if there is an issue

                        # Join the row data into a tab-separated string
                        tsv_row = '\t'.join(row_data)
                        
                        # Write the TSV row to the file
                        f_tsv.write(tsv_row + '\n')
