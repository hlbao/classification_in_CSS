#dictionary-based methods

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

    comment['score'] = score
    if (score > 0):
        comment['sentiment'] = 'positive'
    elif (score < 0):
        comment['sentiment'] = 'negative'
    else:
        comment['sentiment'] = 'neutral'

# Print out summary stats
total = float(len(comments))
num_pos = sum([1 for t in comments if t['sentiment'] == 'positive'])
num_neg = sum([1 for t in comments if t['sentiment'] == 'negative'])
num_neu = sum([1 for t in comments if t['sentiment'] == 'neutral'])
print("Positive: %5d (%.1f%%)" % (num_pos, 100.0 * (num_pos/total)))
print("Negative: %5d (%.1f%%)" % (num_neg, 100.0 * (num_neg/total)))
print("Neutral:  %5d (%.1f%%)" % (num_neu, 100.0 * (num_neu/total)))
