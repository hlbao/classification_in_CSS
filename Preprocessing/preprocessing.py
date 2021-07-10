import requests
import string
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sb
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import Counter
import re
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.stats import ranksums
import time
import multiprocessing as mp
from google.colab import files


url = "https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/historical-users-filtered.json"

#Pull that file and read it into a pandas dataframe
meta_df = pd.read_json(requests.get(url).text)
#meta_df.to_csv('original.csv') 
#files.download('original.csv')

#Download the JSON file; url is defined in the above code block
meta_dict = json.loads(requests.get(url).text)

#Create a receptacle for the accounts' information
meta_data = []

for entity in meta_dict:

  #Pick out only the MoCs
  if entity['type'] == 'member':

    #Take the info we're interested in for each account and add it to our receptacle
    for account in entity['accounts']:
      meta_data.append({'name': entity['name'],
                        'chamber': entity['chamber'],
                        'party': entity['party'],
                        'screen_name': account['screen_name'],
                        'account_type': account['account_type']})
      
#Turn that information into an easy-to-use pandas dataframe
meta_df = pd.DataFrame(meta_data)

#meta_df.to_csv('mete_data.csv') 
#files.download('mete_data.csv')

#The URLs for all the files begin with this path
base_url = "https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/"

#The first and last dates we want tweets for
start_date = "1/1/2020"
end_date = "12/31/2020"

#Add each date to the base URL and make a list of all the resulting file URLs
file_names = []
for d in pd.date_range(start=start_date,end=end_date):
  yr = str(d.year)
  mo = str(d.month).zfill(2) #zfill(2) just makes sure that e.g. "1" gets changed into "01"
  day = str(d.day).zfill(2)
  end = '-'.join([yr,mo,day]) #'-'.join changes e.g. ['2021','06','22'] to "2021-06-22"
  file_names.append(base_url + end + '.json')

#Download all of these files and glue them together into one dataframe
raw_tweet_df = pd.concat([pd.read_json(f) for f in file_names])
#raw_tweet_df
#raw_tweet_df.to_csv('raw_tweet.csv') 
#files.download('raw_tweet.csv')

stops = stopwords.words('english')
stops
#Add tokens to our list of stop words
stops += ['.',',','’',':','&','!','qt','-','"','?','“','”',')','(','/',"'",'–',
          '*',';','‘','>','<']

#Remove tokens from our list of stop words
for w in ['he','we','she','our','if']:
  stops.remove(w)

tknzr = TweetTokenizer()

def clean_tweet(s, stops=stops, tknzr=tknzr):
  #lowercase the string
  s = s.lower()

  #tokenize the string
  s = tknzr.tokenize(s)

  #remove stop words
  s = [w for w in s if w not in stops]

  #remove empty string chunks
  s = [w for w in s if len(w) > 0]

  return s

#Create a new pandas dataframe with only a couple columns of raw_tweet_df
tweet_df = raw_tweet_df[['screen_name','time','text']]

#Remove retweets from this new dataframe
tweet_df = tweet_df[tweet_df.text.apply(lambda x: 'RT @' not in x)]

#Clean the tweets' text
tweet_df.text = tweet_df.text.apply(clean_tweet)

#Merge tweet_df and meta_df
df = tweet_df.merge(meta_df, on="screen_name", how="inner")

#Drop all tweets coming from a MoC who is not a Democrat or Republican
df = df[df.party.apply(lambda x: x in ['D', 'R'])]

#Drop any tweets that were e.g. only stop words
df = df[df.text.apply(lambda x: len(x) > 0)]

#Split up the time column to make it a little easier to use
df['date'] = df.time.apply(lambda x: x.split('T')[0])
df['time'] = df.time.apply(lambda x: x.split('T')[1][:-6])

#Add a column telling us how long each tweet is
df['length'] = df.text.apply(len)
df.to_csv('preprocessed.csv') 
files.download('preprocessed.csv')
