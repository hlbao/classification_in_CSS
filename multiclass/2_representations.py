#should be attached to 1_preprocessing.py
#I separate them for clear illustration reasons.

#two representations: 
#bow (bag of words model)
#it's like a dictionary-based model. Check out this: : https://machinelearningmastery.com/gentle-introduction-bag-words-model/
#the other is tf_idf

count_vect = CountVectorizer() 
count_vect.fit(preprocessed_comments)
final_counts = count_vect.transform(preprocessed_comments)
count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
final_bigram_counts = count_vect.fit_transform(preprocessed_comments)

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
#tf_idf_vect = TfidfVectorizer()
tf_idf_vect.fit(preprocessed_comments)
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)

text_col = ['comment_text']
drop_col = ['id', 'clean','count_sent', 'count_word', 'count_unique_word', 'word_unique_percent']
label_col = [col for col in train_df.columns if col not in text_col + drop_col]
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy
#split
features = text_col
X_test = test_df[features].copy()
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[label_col], test_size=0.2, random_state=2021)
X_train = tf_idf_vect.transform(X_train['comment_text'])
X_val = tf_idf_vect.transform(X_val['comment_text'])
X_test = tf_idf_vect.transform(X_test['comment_text'])
feature_names = tf_idf_vect.get_feature_names()

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
