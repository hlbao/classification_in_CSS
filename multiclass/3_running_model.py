#should be attached to 1_preprocessing.py and 2_representations.py
#I separate them for clear illustration reasons

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
