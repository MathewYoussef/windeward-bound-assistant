import requests
from bs4 import BeautifulSoup

def fetch_webpage_content(url):
    # Fetch the webpage
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve content from {url}")
        return None

    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

# Replace with the URL of the website to scrape
url = "https://www.windewardbound.com.au"
soup = fetch_webpage_content(url)

if soup:
    print("Webpage content fetched successfully.")
    # Display the title to check content parsing
    print("Page Title:", soup.title.string)
