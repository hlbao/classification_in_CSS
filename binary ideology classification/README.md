This is for binary classification with deep learning approaches based on an open-source data set. 

URL = "https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/historical-users-filtered.json."

Created by Honglin Bao, summer 2021. 

I perform partisan inference (democratic vs. republican, binary) based on the Twitter text through NLP techniques. 

Several cleaned Twitter text data sets have been uploaded; see preprocessed.txt for more information (You will mostly play with it). Other data sets are either left unclean (raw tweet.txt) or contain only features for the data set statistics, allowing you to quickly see how this data set looks (e.g., whether the data set is balanced).

I perform a complete process of auto-classification:

1. Pre-processing, lower case, the removal of stop-words, noise, retweet, and punctuation, and so forth.
Note that some steps of pre-processing are not necessarily needed and contribute little to improve the final accuracy.
It really depends on your specific task.
For details, you can check out: https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html
2. load_data for model use
3. feature extraction
4. different representations (vector representations of text, e.g., tf-idf and word2vec) for training/testing/validation data, and the split of training, testing, and validation sets
5. comparing the performance of various binary classification models (e.g., Naive Bayes, Logistic Regression, and SVM) using various evaluation matrices (AUC, confusion matrix, accuracy score, etc.) and testing the trained model

This is the basis for classification and is widely used in computational social science, e.g., semantic analysis.
I also perform advanced classification techniques (e.g., classification with multiple classes or classification using unbalanced/insufficient data). Please take a look at the other folders. 

I discuss only three of the most frequently used classification methods: Naive Bayes, Logistic Regression, and SVM with tf-idf and word2vec vectorizations. Numerous classification methods, such as multilayer perceptron, kNN, and decision tree, can be used here (see https://en.wikipedia.org/wiki/Statistical_classification), but their Python syntax is not significantly different, practically. As a result, I present these three examples just as illustrations. 

Acknowledge: The Summer Institutes in Computational Social Science 2021 (https://sicss.io/)
             Google developer documentation: Text classification (https://developers.google.com/machine-learning/guides/text-classification)


Appreciate and welcome any types of contribution/discussion/pull requests!

Contact: baohlcs@gmail.com
