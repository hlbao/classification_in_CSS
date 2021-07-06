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

url = "https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/historical-users-filtered.json"

#Pull that file and read it into a pandas dataframe
meta_df = pd.read_json(requests.get(url).text)
meta_df

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
meta_df
