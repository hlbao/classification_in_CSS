This is for multi-class classification with machine learning approaches based on a benchmark data set. 

URL = "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"

Created by Honglin Bao, summer 2021. 

Multi-class classification is a slightly more sophisticated classification technique than binary classification:
I want to automatically classify new examples of toxic behavior into one of five categories based on a large number of comments that have been labeled as toxic by human raters (the training set)
: toxic/severe_toxic/obscene/threat/insult/identity_hate.
Toxic Comment, like ideology classification, is also a critical topic in computational social science.

I carry out a complete auto-multiclass-classification procedure:

1. Preprocessing,
converting text to lowercase, 
removing special characters/stop words/permutations, expanding contractions, and so forth. 
"cleaned.jpeg" illustrates the appearance of the preprocessed data set.
Note that certain steps of preprocessing are not always necessary and add little to the final accuracy. 
It truly depends on the nature of the task at hand. 
For additional information, please visit https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html


