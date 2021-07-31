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
tf_idf_vect.fit(preprocessed_comments)
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)

text_col = ['comment_text']
drop_col = ['id', 'clean','count_sent', 'count_word', 'count_unique_word', 'word_unique_percent']
label_col = [col for col in train_df.columns if col not in text_col + drop_col]
final_tf_idf = tf_idf_vect.transform(preprocessed_comments)
