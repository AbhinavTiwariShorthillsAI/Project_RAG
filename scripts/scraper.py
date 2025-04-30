import requests
from bs4 import BeautifulSoup
import time
import random

BASE_URL = "https://en.wikipedia.org"

def scrape_wikipedia_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', {'class': 'mw-parser-output'})
        paragraphs = content_div.find_all('p')

        page_text = ""
        for para in paragraphs:
            text = para.get_text()
            if text.strip():
                page_text += text.strip() + "\n\n"

        return page_text, soup
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

def extract_valid_links(soup):
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/wiki/') and not any(x in href for x in [':', '#']):
            full_link = BASE_URL + href
            links.append(full_link)
    return list(set(links))

def auto_crawl_and_save_one_file(start_urls, max_pages=300, output_file="data/modern_history_combined.txt"):
    visited = set()
    to_visit = list(start_urls)
    full_text = ""

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue

        page_text, soup = scrape_wikipedia_page(url)
        if page_text and soup:
            title = url.split("/wiki/")[-1].replace("_", " ")
            full_text += f"\n\n===== Start of {title} =====\n\n"
            full_text += page_text
            full_text += f"\n===== End of {title} =====\n\n"
            visited.add(url)

            # Expand crawl
            new_links = extract_valid_links(soup)
            random.shuffle(new_links)
            to_visit.extend(new_links)

        time.sleep(1)

    # After crawling, save everything into one big file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Saved all combined text into {output_file} with {len(visited)} pages.")

# -----------------------------
# Start Points for WW1 + WW2 + Related topics
# -----------------------------

start_urls = [
    # World War 1
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/Treaty_of_Versailles",
    "https://en.wikipedia.org/wiki/League_of_Nations",
    "https://en.wikipedia.org/wiki/Battle_of_the_Somme",
    "https://en.wikipedia.org/wiki/Trench_warfare",
    
    # World War 2
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/D-Day",
    "https://en.wikipedia.org/wiki/Holocaust",
    "https://en.wikipedia.org/wiki/Atomic_bombings_of_Hiroshima_and_Nagasaki",
    "https://en.wikipedia.org/wiki/Allies_of_World_War_II",
    
    # Miscellaneous Topics
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/Nuremberg_trials",
    "https://en.wikipedia.org/wiki/Appeasement",
    "https://en.wikipedia.org/wiki/Blitzkrieg",
    "https://en.wikipedia.org/wiki/Operation_Barbarossa"
]

# Start crawling and save into one big file
auto_crawl_and_save_one_file(start_urls, max_pages=300, output_file="data/modern_history_combined.txt")
