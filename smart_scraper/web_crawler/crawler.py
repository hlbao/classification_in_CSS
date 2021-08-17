from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
authors = []
dates = []
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
  #Get the tags and it's class
  statement_footer =  soup.find_all('footer',attrs={'class':'m-statement__footer'})  #Get the tag and it's class
  statement_quote = soup.find_all('div', attrs={'class':'m-statement__quote'}) #Get the tag and it's class
  statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'})#Get the tag and it's class
  target = soup.find_all('div', attrs={'class':'m-statement__meter'}) #Get the tag and it's class
  #loop through the footer class m-statement__footer to get the date and author
  for i in statement_footer:
    link1 = i.text.strip()
    name_and_date = link1.split()
    first_name = name_and_date[1]
    last_name = name_and_date[2]
    full_name = first_name+' '+last_name
    month = name_and_date[4]
    day = name_and_date[5]
    year = name_and_date[6]
    date = month+' '+day+' '+year
    dates.append(date)
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

#From October 22, 2009 to August 12, 2021
n=21069
for i in range(1, n+1):
  scrape_website(i)
#Create a new dataFrame 
data = pd.DataFrame(columns = ['author',  'statement', 'source', 'date', 'target']) 
data['author'] = authors
data['statement'] = statements
data['source'] = sources
data['date'] = dates
data['target'] = targets

data.to_csv('political_fact_dataset.csv')
