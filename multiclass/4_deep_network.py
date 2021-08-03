import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objs as go
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#run deep learning
#should be attached to 2_representations.py
#I separate them for clear illustration reasons.

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each comment.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[label_col], test_size=0.2, random_state=2021)
X_train = tf_idf_vect.transform(X_train['comment_text'])
X_val = tf_idf_vect.transform(X_val['comment_text'])
X_test = tf_idf_vect.transform(X_test['comment_text'])
feature_names = tf_idf_vect.get_feature_names()

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=train_df[features].shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
epochs = 5
batch_size = 64

#converting your labels to arrays before calling model.fit()
X_train= np.array(X_train, dtype=object)
y_train= np.array(y_train, dtype=object)
X_val = np.array(X_val, dtype=object)
y_val = np.array(y_val, dtype=object)

lstm_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.0,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,y_test)
#note: if you are working on TensorFlow 2.1.0, the above converting code will give you an error, i.e., 
#ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float) (located at the line of model.fit())
#I run my code on TensorFlow 2.0.0.beta1
