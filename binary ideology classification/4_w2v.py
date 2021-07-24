#word to vector
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embeddin
import gensim
from gensim.models import Word2Vec #Word2Vec is mostly used for huge datasets
#you can download the data from https://www.dropbox.com/s/o715p3dsr55cxhc/preprocessed.csv?dl=0
import os
os.chdir('/Users/honglinbao/Desktop/')
#your own path

df_train=pd.read_csv('preprocessed.csv')
for i in df_train["party"]:
    if (i =="R"):
        df_train['new_label'][i] ==1
    else:
        df_train['new_label'][i] ==0
    
df_train['clean_text_tok']=[nltk.word_tokenize((' '.join(i)) for i in df_train['text'])] 
model = Word2Vec(df_train['clean_text_tok'],min_count=1) 
w2v = dict(zip(model.wv.index2word, model.wv.syn0))  #combination of word and its vector

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
      
      
#SPLITTING THE TRAINING DATASET INTO TRAINING AND VALIDATION

X_train, X_val, y_train, y_val = train_test_split(df_train["text"],
                                                  df_train["new_label"],
                                                  test_size=0.3,
                                                  shuffle=True)

X_train_tok= [nltk.word_tokenize(i) for i in X_train]  #for word2vec
X_val_tok= [nltk.word_tokenize(i) for i in X_val]      #for word2vec

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val) 

# Fit and transform
modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_val_tok)
