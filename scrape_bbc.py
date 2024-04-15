from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from tqdm import tqdm

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


    all_cards = []
    for _ in tqdm(range(7), desc="Loading news"):
        click_load_more(driver)

        # Wait for new elements to be loaded; adjust the condition based on actual content
        time.sleep(3)  # Adjust this time based on trial and error to find the minimum stable waiting time
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[@data-testid='liverpool-card']"))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        liverpool_cards = soup.find_all(attrs={"data-testid": "liverpool-card"})
        all_cards.extend(liverpool_cards)
        print(f"Cards found this iteration: {len(liverpool_cards)}")

    driver.quit()
    return all_cards, len(all_cards)

def click_load_more(driver):
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='pagination-next-button']"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
        load_more_button.click()
    except TimeoutException:
        print("Timeout waiting for the 'Load More' button to become clickable.")
    except NoSuchElementException:
        print("Load More button not found.")
    except Exception as e:
        print(f"Failed to click Load More: {e}")

cards, counter = scrape_news_business()
print(f"Found {counter} Liverpool cards")

# Save the cards to a file
with open("liverpool_cards.txt", "w") as file:
    for card in cards:
        file.write(str(card) + "\n")
