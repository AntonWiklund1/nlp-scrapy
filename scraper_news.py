from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import requests_cache
import csv
import pandas as pd
from datetime import datetime, timedelta

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
    time.sleep(0.5)  # Give the page a moment to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the article headline
    headline = soup.find('h1')
    if headline:
        headline = headline.text.strip()
    else:
        headline = "No headline found"

    # Find the article's publication time
    time_elem = soup.find('time')
    if time_elem:
        time_elem = time_elem.text.strip()
    else:
        time_elem = "No time found"

    # Locate the <article> tag and extract all <p> tags within it
    article = soup.find('article')
    if article:
        paragraphs = article.find_all('p')
        body_text = ' '.join(paragraph.get_text(separator=' ', strip=True) for paragraph in paragraphs)
    else:
        body_text = "No body found"

    return {
        "headline": headline,
        "body": body_text,
        "time": time_elem,
        "url": url,
        "Category": category
    }

# Define the standardize_date function that will be used later
def standardize_date(date_str):
    current_date = datetime.now()
    if 'ago' in date_str:
        number, unit = date_str.split()[:2]
        number = int(number)
        if 'day' in unit:
            return current_date - timedelta(days=number)
        elif 'hour' in unit:
            return current_date - timedelta(hours=number)
        elif 'minute' in unit:
            return current_date - timedelta(minutes=number)
    elif 'Just now' in date_str:
        return current_date
    else:
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except ValueError:
            return pd.NaT  # Not a Time for unparseable formats

def click_load_more(driver):
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='pagination-next-button']"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
        load_more_button.click()
    except Exception as e:
        print(f"Failed to click Load More: {e}")

def save_to_csv_by_day(data):
    grouped = data.groupby('time')  # Group the articles by the 'time' column
    total_lenght = len(data)
    for date, group in grouped:
        filename = f'./data/scraped_by_day/articles_{date}.csv'  # Naming the file with the respective date
        group.to_csv(filename, index=False)  # Save each group to a CSV file without the index
        print(f"Saved {len(group)} articles to {filename}")
    print(f"Total articles saved: {total_lenght}")
 
#clear ./data/scraped_by_day folder
import os
import shutil
folder = './data/scraped_by_day'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

current_date = datetime.now()

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

filtered_articles = pd.DataFrame(filtered_articles)

# Apply the standardization function to the 'time' column
filtered_articles['time'] = filtered_articles['time'].apply(standardize_date)

# Formatting to YYYY-MM-DD if needed
filtered_articles['time'] = filtered_articles['time'].dt.strftime('%Y-%m-%d')

# Write the DataFrame to a CSV file
save_to_csv_by_day(filtered_articles)

filtered_articles.to_csv(f'./data/scraped/all_articles.csv', index=False)
