This is for multi-class classification with machine learning approaches based on a benchmark data set on Kaggle.  

URL = "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"

Created by Honglin Bao, summer 2021. 

Multi-class classification is a slightly more sophisticated classification technique than binary classification:
I want to automatically classify new examples of toxic behavior into one of five categories based on a large number of comments that have been labeled as toxic by human raters (the training set)
: toxic/severe_toxic/obscene/threat/insult/identity_hate.
As with ideology classification, the study of toxic comments is also a critical topic in computational social science.

I carry out a complete auto-multiclass-classification procedure:

1. Preprocessing,
converting text to lowercase, 
removing special characters/html/stop words/punctuation, expanding contractions, and so forth. 
"cleaned.jpeg" illustrates the appearance of the preprocessed data set.
Additionally, the data set of cleaned comment text has been uploaded. Please refer to cleaned.txt


Note that certain steps of preprocessing are not always necessary and add little to the final accuracy. 
It truly depends on the nature of the task at hand. 
For additional information, please visit https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html

2. different representations (vector representations of text, e.g., tf-idf and bag-of-words), the split of training, testing, and validation sets, and feature extraction

3. comparing the performance of various multi-class classification models (e.g., Multinomial Naive Bayes, Logistic Regression, RandomForest, and Support Vector Classification) using the AUC evaluation matrix and testing your trained model. The average area under the ROC curve (ROC-AUC) is around 95%

Additionally, I perform several multi-label classification methods, including Label Powerset, Binary Relevance, and Classifier Chain. This is because latent connections exist between multi-class and multi-label classification methods: "For each label combination in the training set, the label powerset transformation generates a binary classifier. For instance, if the possible labels for an example are A, B, and C, the label powerset representation of this problem is a multi-class classification problem with the classes [0 0 0], [1 0 0], [0 1 0], [0 0 1], [1 0 1], [0 1 1]. [1 1 1], where [1 0 1] denotes an instance in which labels A and C are present but label B is not." See https://www.sciencedirect.com/science/article/pii/S1571066113000121?via%3Dihub This is also very similar to the process of multi-label classification on the same data set. Susan Li has an established model for multi-label classification — a text data example sharing multiple labels such as severe toxic, threats, obscenity, and insults. Consider the following: https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

Similarly, numerous classification methods can be used here for multiclass/multilabel classification, but their Python syntax is not significantly different, practically. As a result, I present certain examples just as illustrations.

4. deep learning-based approach. Besides using embedded, traditional machine learning methods, we can also use fancy deep learning techniques to perform this multi-class classification. I choose an LSTM (Long short-term memory)-based neural network architecture, which is widely used in multi-class classification. A bit different step is "Tokenizer" -- it acts like "vectorization," preparing the text inputs for a model as "tokens," i.e., an instance of a sequence of characters that are grouped together as a functional semantic unit for processing, then putting these tokens being represented by vector embeddings (e.g., one-hot) into an embedding layer to map the text data into a proper dimensional setting, then LSTM for classification with dropout, and finally dense layer with the required number of classes. The activation function is softmax. The loss function is categorical_crossentropy. "LSTM.jpeg" illustrates the network architecture

5. Based on the cleaned social media comment data set, I also perform a dictionary-based method for sentiment multi-class classification (positive, negative, or neutral). The fundamental concept underlying the lexicon-based classification model can be found at https://methods.sagepub.com/dataset/howtoguide/dictionary-based-sentiment-analysis-in-econnews-2016-python. The required dataset has been uploaded, as well as the associated result. The file lexicon_table.csv contains a table of frequently used positive and negative words and their scores. Typically, lexicon-based models perform poorly compared to machine/deep learning-based models and have many limitations. Please refer to this post (https://dzone.com/articles/reasons-to-replace-dictionary-based-text-mining-wi) for details.

Note that this is the most basic multi-layer neural network architecture for multi-class classification with decent accuracy. There are many fancy variations, like bert, GPT, transformer, and attention model, with very high accuracy and dealing with many real-world limitations of the basic model, e.g., insufficient/imbalanced data. Please check other folders for details.

Appreciate and welcome any types of contribution/discussion/pulling requests!

Contact: baohlcs@gmail.com
