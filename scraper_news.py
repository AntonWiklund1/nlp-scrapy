from bs4 import BeautifulSoup
import pandas as pd

import requests

url = 'https://www.nbcnews.com/world'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

news = soup.find_all('div', class_='wide-tease-item__wrapper df flex-column flex-row-m flex-nowrap-m')

#get the headline
headlines = [item.find('h2', class_='wide-tease-item__headline').text for item in news]

links = [item.find('div', class_='wide-tease-item__image-wrapper flex-none relative dt dn-m').a['href'] for item in news]

dates = []
bodies = []
for link in links:
    try:
        article_response = requests.get(link)
        article_soup = BeautifulSoup(article_response.text, 'html.parser')
        time_element = article_soup.find('time', class_='relative z-1')
        if time_element and 'datetime' in time_element.attrs:
            date = time_element['datetime']
        else:
            date = 'Date not found'
        
        body_element = article_soup.find('div', class_='article-body__content')
        if body_element:
            body = body_element.text
        else:
            body = 'Body not found'
        
    except Exception as e:
        print(f"Error fetching article date: {e}")
        date = 'Error fetching date'
    dates.append(date)
    bodies.append(body)



# Generate IDs
ids = range(1, 1 + len(headlines))

# Directly create the DataFrame with columns in the desired order
df = pd.DataFrame({
    'id': ids,
    'headline': headlines,
    'link': links,
    'date': dates,
    'body': bodies
})

print(df)

#export the dataframe to a txt file
df.to_csv('nbcnews.txt', index=False)
