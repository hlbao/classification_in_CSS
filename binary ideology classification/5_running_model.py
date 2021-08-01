#should be attached to 4_w2v.py
#as step 4 is a word-to-vector process. Step 5 is running different binary classification models.

# what should be chosen as the evaluation matrix?
#walk through this page: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
#as well as this Wikipedia page: https://en.wikipedia.org/wiki/Binary_classification

#FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
lr_tfidf=LogisticRegression(solver = 'liblinear', C=1.0, penalty = 'l2')
#lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  #model
y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
#the best
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_vectors_tfidf, y_train)  #model
y_predict = nb_tfidf.predict(X_val_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_val_vectors_tfidf)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using SVM (tf_idf)
SVM_tfidf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM_tfidf.fit(X_train_vectors_tfidf, y_train)
y_predict = SVM_tfidf.predict(X_val_vectors_tfidf)
y_prob = SVM_tfidf.predict_proba(X_val_vectors_tfidf)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y_predict, y_val)*100)

#FITTING THE CLASSIFICATION MODEL using Logistic Regression (w2v)
lr_w2v=LogisticRegression(solver = 'liblinear', C=1.0, penalty = 'l2')
lr_w2v.fit(X_train_vectors_w2v, y_train)  #model
y_predict = lr_w2v.predict(X_val_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_val_vectors_w2v)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using Naive Bayes(w2v)
nb_w2v = MultinomialNB()
nb_w2v.fit(X_train_vectors_w2v, y_train)  #model
y_predict = nb_w2v.predict(X_val_vectors_w2v)
y_prob = nb_w2v.predict_proba(X_val_vectors_w2v)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using SVM (w2v)
SVM_w2v = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM_w2v.fit(X_train_vectors_w2v, y_train)
# predict the labels on validation dataset
y_predict = SVM_w2v.predict(X_val_vectors_w2v)
y_prob = SVM_w2v.predict_proba(X_val_vectors_w2v)[:,1]
print(classification_report(y_val,y_predict))
print('Confusion Matrix:',confusion_matrix(y_val, y_predict))
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y_predict, y_val)*100)

#you can choose a random part of the preprocessed table as the test set 
#for sure, remove the 'party' column
#or you can keep the 'party' column to see how accurate your model is 
#name it as test.csv with the same names for all columns
df_test=pd.read_csv('test.csv')  
df_test['test_set']=[(' '.join(i)) for i in df_test['text']] 
X_test=df_test['test_set']
X_vector=tfidf_vectorizer.transform(X_test) #converting X_test to vector
y_predict = lr_tfidf.predict(X_vector)      #use the trained model on X_vector
y_prob = lr_tfidf.predict_proba(X_vector)[:,1]
#change lr_tfidf to any model you want to test: SVM_tfidf, nb_tfidf, nb_w2v, SVM_w2v, and lr_w2v.
df_test['predict_prob']= y_prob
df_test['result']= y_predict
#print(df_test.head())
df_test.to_csv('your_final_prediction.csv')


