from bs4 import BeautifulSoup

import requests

url = 'https://www.nbcnews.com/world'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

news = soup.find_all('div', class_='wide-tease-item__wrapper df flex-column flex-row-m flex-nowrap-m')

#get the headline
headlines = [item.find('h2', class_='wide-tease-item__headline').text for item in news]

link = [item.find('div', class_='wide-tease-item__image-wrapper flex-none relative dt dn-m').a['href'] for item in news]

print('NBC News')
print('---------------------------------')
for headline, link in zip(headlines, link):
    print(f'Headline: {headline}')
    print(f'Link: {link}')
    print('---------------------------------')