from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import requests_cache
import csv

# Set up caching
requests_cache.install_cache('bbc_cache', backend='sqlite', expire_after=300)

def scrape_news(category):
    options = webdriver.ChromeOptions()
    # Uncomment the next line to run in headless mode once debugging is complete
    # options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(f'https://www.bbc.com/{category}')


    time.sleep(3)  # Wait for the page to load
    WebDriverWait(driver, 10).until(
        EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='SP Consent Message']"))
    )
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I agree')]"))
    ).click()
    driver.switch_to.default_content()

    all_articles = []
    base_url = "https://www.bbc.com"  # Base URL for constructing full URLs from relative paths
    urls = []
    for _ in tqdm(range(9), desc=f"Loading news for {category}"):
        click_load_more(driver)
        time.sleep(3)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[@data-testid='liverpool-card']"))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        liverpool_cards = soup.find_all(attrs={"data-testid": "liverpool-card"})
        
        for card in liverpool_cards:
            link = card.find('a', attrs={"data-testid": "internal-link"})
            if link and link.has_attr('href'):
                url = link['href']
                if not url.startswith('http'):
                    url = base_url + url  # Ensure correct URL formation
                urls.append(url)
                #detailed_content = scrape_detailed_page(driver, url)
                #all_articles.append(detailed_content)

    driver.quit()
    print(f"Found {len(urls)} articles for {category}")
    return urls

def scrape_detailed_page(driver, url, category):
    driver.get(url)
    time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    headline = soup.find('h1')
    sections = soup.find_all('section', attrs={"data-component": "text-block"})
    time_elem = soup.find('time')
    
    # Extracting text from each section and concatenating it
    body_text = ' '.join(section.get_text(separator=' ', strip=True) for section in sections)

    return {
        "headline": headline.text.strip() if headline else "No headline found",
        "body": body_text if body_text else "No body found",
        "time": time_elem.text.strip() if time_elem else "No time found",
        "url": url,
        "Category": category
    }



def click_load_more(driver):
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='pagination-next-button']"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
        load_more_button.click()
    except Exception as e:
        print(f"Failed to click Load More: {e}")

categories = [
    'innovation',
    'business',
    'culture',
    'news/topics/c2vdnvdg6xxt',  # Israel Gaze
    'news/war-in-ukraine'        # Ukrainian News
]

category_urls = {}

#scrape the detailed content
articles = []
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

for category in tqdm(categories, desc="Scraping categories"):
    # Rename specific categories to 'politics'
    if category in ['news/topics/c2vdnvdg6xxt', 'news/war-in-ukraine']:
        category_label = 'politics'
    elif category == 'innovation':
        category_label = 'tech'
    elif category == 'culture':
        category_label = 'entertainment'
    else:
        category_label = category
    
    category_urls[category] = scrape_news(category)
    for url in tqdm(category_urls[category], desc=f"Scraping articles for {category_label}"):
        detailed_content = scrape_detailed_page(driver, url, category_label)
        articles.append(detailed_content)

driver.quit()

filtered_articles = [article for article in articles if article['body'] != "No body found" and article['time'] != "No time found"]

with open("./data/processed/bbc_articles.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["id", "url", "headline", "body", "time", "Category"])
    writer.writeheader()
    for i, article in enumerate(filtered_articles, 1):
        writer.writerow({"id": i, **article})

print(f"Saved {len(filtered_articles)} articles to './data/processed/bbc_articles.csv'")
