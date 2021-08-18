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
#should be attached to 3_machine_learning_models.py
#I separate them for clear illustration reasons.

#re-organize the data set
sub_df_rf.iloc[:,1:] = preds_test
sub_df_rf.to_csv('submission_rf.csv')
uploaded = files.upload()
final_submission_combined_updated = pd.read_csv('sample_submission.csv',error_bad_lines=False, engine="python")
for label in label_col:
    final_submission_combined_updated[label] = 0.5*(sub_df_mnb[label]+sub_df_lr[label])
final_submission_combined_updated.to_csv('final_submission_combined_updated.csv', index=False)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each comment.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_df[features].values)

X = tokenizer.texts_to_sequences(train_df[features].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(train_df[label_col]).values
#one-hot encoding

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=2021)
#X_train = tf_idf_vect.transform(X_train['comment_text'])
#X_val = tf_idf_vect.transform(X_val['comment_text'])
#X_test = tf_idf_vect.transform(X_test['comment_text'])
#feature_names = tf_idf_vect.get_feature_names()

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length = X.shape[1]))
#if you are working with the LSTM network, then you should first have an embedding layer
#the purpose is to map your data into a proper dimensional setting before calling model.fit()
#model.add(Dense(6, activation='relu', input_dim = X_train.shape[1])) will also help you to achieve that
#input_length is the length of the input text data
#input_dim is the dimension of the text data. 
#For example, if the content of a piece of text data is: ['apple', 'apple', 'car']
#one-hot encoding will be [[1 0], [1 0], [0 1]]. batch_size = 3, input_dim = 2, input_length = 3.
model.add(SpatialDropout1D(0.2))
#model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

epochs = 5
batch_size = 64

#you might need this (line 86-89) -- depending on which version of TensorFlow you are working with
#converting your labels to arrays before calling model.fit()
#if you are working on TensorFlow 2.1.0, the following converting code will give you an ERROR, i.e., 
#ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float) (located at the line of model.fit())
#I run my code on TensorFlow 2.0.0.beta1
#X_train = np.asarray(X_train).astype('float32')
#y_train = np.asarray(y_train).astype('float32')
#X_val = np.asarray(X_val).astype('float32')
#y_val = np.asarray(y_val).astype('float32')

#lstm_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
lstm_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.0,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_val,y_val)
