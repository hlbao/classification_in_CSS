import csv
import re
import operator
import pandas as pd

comments = []

from google.colab import files
uploaded = files.upload()
#new_cleaned.csv just includes the "comment_text" column
#you can access this dataset on my Github (check out new_cleaned.txt)
filename = 'new_cleaned.csv'
with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
      comments.append(row[2])
  
#print(comments)

lexicon = dict()
from google.colab import files
uploaded = files.upload()
filename = 'lexicon_table.csv'
with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
      lexicon[row[0]] = int(row[1])
  
# Use lexicon to score comments
for comment in comments:
    score = 0
    for word in comment.split():
        if word in lexicon:
            score = score + lexicon[word]
    if (score > 0):
        print('positive')
    elif (score < 0):
        print('negative')
    else:
        print('neutral')
