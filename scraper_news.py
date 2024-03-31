import time
from bs4 import BeautifulSoup
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

def scrape_news(continent):
    driver = webdriver.Chrome()
    driver.get(f'https://www.nbcnews.com/news/{continent}')

    # Click the "Load More" button a specified number of times
    for _ in tqdm(range(7), desc=f"Loading news for {continent}"):
        try:
            load_more_button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//button[@data-testid='button-hover-animation']"))
            )
            # Scroll the button into view
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            load_more_button.click()
            time.sleep(0.5)  # Wait for the page to load
        except Exception as e:
            print(f"Failed to click Load More: {e}")
            break

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # Clean up and close the browser
    driver.quit()

    return soup

def handle_soup(soup, continent = 'unspecified'):
    # Find all news items
    news_items = soup.find_all('div', class_='wide-tease-item__wrapper df flex-column flex-row-m flex-nowrap-m')

    # Get headlines
    headlines = [item.find('h2', class_='wide-tease-item__headline').text.strip() for item in news_items]

    # Extract the links and handle cases where the expected structure is not found
    links = []
    for item in news_items:
        div = item.find('div', class_='wide-tease-item__image-wrapper flex-none relative dt dn-m')
        link = div.a.get('href') if div and div.a else None
        links.append(link)

    # Initialize lists for dates and bodies
    dates = []
    bodies = []

    # Go through each link to extract dates and bodies
    for link in tqdm(links, desc=f"Fetching articles for {continent}"):
        if link:
            try:
                article_response = requests.get(link)
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                time_element = article_soup.find('time', class_='relative z-1')
                date = time_element['datetime'] if time_element and 'datetime' in time_element.attrs else 'Date not found'
                
                body_element = article_soup.find('div', class_='article-body__content')
                body = body_element.text.strip() if body_element else 'Body not found'
                
            except Exception as e:
                print(f"Error fetching article data: {e}")
                date = 'Error fetching date'
                body = 'Error fetching body'
        else:
            date = 'No link found'
            body = 'No link found'
            
        dates.append(date)
        bodies.append(body)
    
    return pd.DataFrame({
        'headline': headlines,
        'link': links,
        'date': dates,
        'body': bodies
    })

europe_soup = scrape_news('europe')
us_soup = scrape_news('us-news')
africa_soup = scrape_news('africa')

europe_df = handle_soup(europe_soup, 'europe')
us_df = handle_soup(us_soup, 'us')
africa_df = handle_soup(africa_soup, 'africa')

completed_df = pd.concat([europe_df, us_df, africa_df], ignore_index=True, axis=0)


print("Done scraping news!")
print(completed_df.head())

# Export the DataFrame to a .txt file
completed_df.to_csv('nbcnews.csv', sep=",", index=True)
