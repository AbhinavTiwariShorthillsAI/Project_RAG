import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List

BASE_URL = "https://en.wikipedia.org"

def scrape_wikipedia_page(url: str) -> tuple[str, BeautifulSoup] | tuple[None, None]:
    """
    Scrapes the main paragraph content from a given Wikipedia page.

    Args:
        url (str): Full URL of the Wikipedia page.

    Returns:
        tuple[str, BeautifulSoup] | tuple[None, None]: Tuple containing the extracted text and parsed soup,
        or (None, None) on error.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Will raise HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', {'class': 'mw-parser-output'})
        paragraphs = content_div.find_all('p')

        page_text = ""
        for para in paragraphs:
            text = para.get_text()
            if text.strip():
                page_text += text.strip() + "\n\n"

        return page_text, soup
    except requests.exceptions.RequestException as e:
        # Handle network errors, invalid URL, etc.
        print(f"Error requesting {url}: {e}")
        return None, None
    except Exception as e:
        # Handle other unforeseen exceptions
        print(f"Error scraping {url}: {e}")
        return None, None

def extract_valid_links(soup: BeautifulSoup) -> List[str]:
    """
    Extracts valid internal Wikipedia article links from a page soup.

    Args:
        soup (BeautifulSoup): Parsed HTML of the Wikipedia page.

    Returns:
        List[str]: List of full Wikipedia article URLs.
    """
    links = []
    try:
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/wiki/') and not any(x in href for x in [':', '#']):
                full_link = BASE_URL + href
                links.append(full_link)
        return list(set(links))
    except Exception as e:
        # Handle any issues while extracting links
        print(f"Error extracting links: {e}")
        return []

def auto_crawl_and_save_one_file(
        start_urls: List[str], 
        max_pages: int = 300, 
        output_file: str = "data/modern_history_combined.txt"
        ) -> None:
    """
    Automatically crawls Wikipedia articles starting from seed URLs and stores the content in one file.

    Args:
        start_urls (List[str]): List of Wikipedia URLs to start crawling from.
        max_pages (int): Maximum number of pages to crawl.
        output_file (str): Path to the file where the combined output is saved.
    """
    visited = set()
    to_visit = list(start_urls)
    full_text = ""

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue

        page_text, soup = scrape_wikipedia_page(url)
        if page_text and soup:
            try:
                title = url.split("/wiki/")[-1].replace("_", " ")
                full_text += f"\n\n===== Start of {title} =====\n\n"
                full_text += page_text
                full_text += f"\n===== End of {title} =====\n\n"
                visited.add(url)

                new_links = extract_valid_links(soup)
                random.shuffle(new_links)
                to_visit.extend(new_links)
            except Exception as e:
                # Handle any issues while processing the page content
                print(f"Error processing content from {url}: {e}")
        else:
            print(f"Skipping {url} due to errors while scraping.")

        time.sleep(random.uniform(1, 3))  # Randomized delay to avoid being blocked

    try:
        if full_text.strip():  # Only write if full_text is not empty or just whitespace
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"✅ Saved all combined text into {output_file} with {len(visited)} pages.")
        else:
            print(f"❌ No pages were saved due to errors.")
    except Exception as e:
    # Handle any errors while writing to the file
        print(f"Error saving the file {output_file}: {e}")


if __name__ == "__main__":
    start_urls = [
        "https://en.wikipedia.org/wiki/World_War_I",
        "https://en.wikipedia.org/wiki/Treaty_of_Versailles",
        "https://en.wikipedia.org/wiki/League_of_Nations",
        "https://en.wikipedia.org/wiki/Battle_of_the_Somme",
        "https://en.wikipedia.org/wiki/Trench_warfare",
        "https://en.wikipedia.org/wiki/World_War_II",
        "https://en.wikipedia.org/wiki/D-Day",
        "https://en.wikipedia.org/wiki/Holocaust",
        "https://en.wikipedia.org/wiki/Atomic_bombings_of_Hiroshima_and_Nagasaki",
        "https://en.wikipedia.org/wiki/Allies_of_World_War_II",
        "https://en.wikipedia.org/wiki/Cold_War",
        "https://en.wikipedia.org/wiki/Nuremberg_trials",
        "https://en.wikipedia.org/wiki/Appeasement",
        "https://en.wikipedia.org/wiki/Blitzkrieg",
        "https://en.wikipedia.org/wiki/Operation_Barbarossa"
    ]

    auto_crawl_and_save_one_file(start_urls)
