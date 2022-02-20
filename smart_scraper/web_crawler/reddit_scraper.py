import pprint
import pandas as pd
import numpy as np
import datetime
from bs4 import BeautifulSoup
import requests
import time
import os
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15'}

url="https://www.reddit.com/r/redditlists/comments/josdr/list_of_political_subreddits/"
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
req = Request(url , headers={'User-Agent': 'Mozilla/5.0'})

webpage = urlopen(req).read()
page_soup = soup(webpage, "html.parser")
title = page_soup.find("title")
print(title)
containers = page_soup.findAll('a',class_="_3t5uN8xUmg0TOwRCOGQEcU")

names=[]
urls=[]
pairs=[]

for container in containers[:174]:
    names.append(container.text)
    urls.append(container['href'])
    pairs.append((container.text,container['href']))
    print(container.text)
    print(container['href'])
