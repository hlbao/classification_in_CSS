from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
authors = []
statements = []
sources = []
targets = []

#Create a function to scrape the site
def scrape_website(page_number):
  page_num = str(page_number) #Convert the page number to a string
  URL = 'https://www.politifact.com/factchecks/list/?page='+page_num #append the page number to complete the URL
  webpage = requests.get(URL)  #Make a request to the website
  #time.sleep(3)
  soup = BeautifulSoup(webpage.text, "html.parser") #Parse the text from the website
  statement_footer =  soup.find_all('footer',attrs={'class':'m-statement__footer'})  #Get the tag and it's class
  statement_quote = soup.find_all('div', attrs={'class':'m-statement__quote'}) #Get the tag and it's class
  statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'})#Get the tag and it's class
  target = soup.find_all('div', attrs={'class':'m-statement__meter'}) #Get the tag and it's class
  #loop through the footer class m-statement__footer to get the date and author
  for i in statement_footer:
    link1 = i.text.strip()
    name_full = link1.split()
    first_name = name_full[1]
    last_name = name_full[2]
    full_name = first_name+' '+last_name
    authors.append(full_name)
  #Loop through the div m-statement__quote to get the link
  for i in statement_quote:
    link2 = i.find_all('a')
    statements.append(link2[0].text.strip())
 #Loop through the div m-statement__meta to get the source
  for i in statement_meta:
    link3 = i.find_all('a') #Source
    source_text = link3[0].text.strip()
    sources.append(source_text)
  #Loop through the target or the div m-statement__meter to get the facts about the statement (True or False)
  for i in target:
    fact = i.find('div', attrs={'class':'c-image'}).find('img').get('alt')
    targets.append(fact)

n=100
for i in range(1, n+1):
  scrape_website(i)
data = pd.DataFrame(columns = ['author',  'statement', 'source', 'target']) 
data['author'] = authors
data['statement'] = statements
data['source'] = sources
data['target'] = targets

data.to_csv('political_fact_dataset.csv')
from google.colab import files
files.download("political_fact_dataset.csv")
