from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from tqdm import tqdm
import requests_cache

def scrape_news_business():
    options = webdriver.ChromeOptions()
    # Uncomment the next line to run in headless mode once debugging is complete
    # options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get('https://www.bbc.com/business')

    WebDriverWait(driver, 10).until(
        EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='SP Consent Message']"))
    )
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I agree')]"))
    ).click()
    driver.switch_to.default_content()

    all_articles = []
    base_url = "https://www.bbc.com"  # Base URL for constructing full URLs from relative paths

    for _ in tqdm(range(7), desc="Loading news"):
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
                detailed_content = scrape_detailed_page(driver, url)
                all_articles.append(detailed_content)

    driver.quit()
    return all_articles, len(all_articles)

def scrape_detailed_page(driver, url):
    print("Navigating to:", url)  # Debug print to check what URL is being loaded
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    headline = soup.find('h1')
    body = soup.find('div', attrs={"data-component": "text-block"})
    time_elem = soup.find('time')
    
    return {
        "headline": headline.text if headline else "No headline found",
        "body": body.text if body else "No body found",
        "time": time_elem.text if time_elem else "No time found"
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


articles, count = scrape_news_business()
print(f"Found {count} articles")

# Save the articles to a file
with open("bbc_articles.txt", "w") as file:
    for article in articles:
        file.write(f"Headline: {article['headline']}, Body: {article['body']}, Time: {article['time']}\n")
