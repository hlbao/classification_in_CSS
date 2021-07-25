This is for binary classification with deep learning approaches based on an open-source data set URL = "https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/historical-users-filtered.json."
Created by Honglin Bao, summer 2021. 

I perform partisan inference (democratic vs. republican, binary) based on Twitter text through NLP techniques.

I perform a complete process of auto-classification:

1. Pre-processing, lower case, the removal of stop-words, noise, retweet, and punctuation, and so forth.
Note that some steps of pre-processing are not necessarily needed and contribute little to improve the final accuracy.
It really depends on your specific task.
For details, you can check out: https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html
2. load_data for model use
3. feature extraction
4. different representations (vector representations of text, e.g., tf-idf and word2vec) for training/testing data, and the split of training, testing, and validation sets
5. running different binary classification models (e.g., Naive Beyas and Logistic Regression) with different evaluation matrices (AUC, confusion_matrix, etc.)

This is the basis for classification and is widely used in computational social science, e.g., semantic analysis.
I also perform advanced classification techniques (e.g., multi-class classification or classification with unbalanced/insufficient data). Please check out other folders.

Acknowledge: The Summer Institutes in Computational Social Science 2021 (https://sicss.io/)
             Google developer documentation: text classification (https://developers.google.com/machine-learning/guides/text-classification)


Appreciate and welcome any types of contribution/discussion/pull requests!
Contact: baohlcs@gmail.com

