#should be attached to 1_preprocessing.py and 2_representations.py
#I separate them for clear illustration reasons.

from google.colab import files
uploaded = files.upload()
sub_df_rf = pd.read_csv('sample_submission.csv')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

model = RandomForestClassifier()
#model = MultinomialNB(alpha = 0.1)
#model = LogisticRegression(C=10)
#model = BinaryRelevance(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),require_dense=[False, True])
#model = LabelPowerset(LogisticRegression(),require_dense = [False, True])
#model = ClassifierChain(MultinomialNB(),require_dense = [False, True])
#model = BinaryRelevance(SVC(C=1.0, probability=True), require_dense = [False, True])

train_rocs = []
valid_rocs = []
preds_train = np.zeros(y_train.shape)
preds_valid = np.zeros(y_val.shape)
preds_test = np.zeros((len(test_df), len(label_col)))

for i, label_name in enumerate(label_col):
    print('\nClass:= '+label_name)
    # fit
    model.fit(X_train,y_train[label_name])
    # train
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_roc_class = roc_auc_score(y_train[label_name],preds_train[:,i])
    print('Train ROC AUC:', train_roc_class)
    train_rocs.append(train_roc_class)
    # valid
    preds_valid[:,i] = model.predict_proba(X_val)[:,1]
    valid_roc_class = roc_auc_score(y_val[label_name],preds_valid[:,i])
    print('Valid ROC AUC:', valid_roc_class)
    valid_rocs.append(valid_roc_class)
    # test predictions
    preds_test[:,i] = model.predict_proba(X_test)[:,1]
    
print(np.mean(train_rocs))
print(np.mean(valid_rocs))

#process text set and test your trained model
from google.colab import files
uploaded = files.upload()
test_df=pd.read_csv('test.csv')  
test_df['comment_text'] = test_df['comment_text'].str.lower()
test_df['comment_text'] = test_df['comment_text'].apply(cleanHtml)
test_df['comment_text'] = test_df['comment_text'].apply(cleanPunc)
test_df['comment_text'] = test_df['comment_text'].apply(keepAlpha)
test_df['comment_text'] = test_df['comment_text'].apply(removeStopWords)
import itertools
from bs4 import BeautifulSoup
from tqdm import tqdm
preprocessed_comments = []
for sentence in tqdm(test_df['comment_text'].values):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))
    preprocessed_comments.append(sentence.strip())   
test_df.to_csv('test_cleaned.csv') 
files.download('test_cleaned.csv')
uploaded = files.upload()

df_test=pd.read_csv('test_cleaned.csv')  
X_test=df_test['comment_text']
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_vector= tfidf_vectorizer.transform(X_test) #converting X_test to vector
y_predict = model.predict(X_vector)      #use the trained model on X_vector
y_prob = model.predict_proba(X_vector)[:,1]
df_test['predict_prob']= y_prob
df_test['result']= y_predict
#print(df_test.head())
df_test.to_csv('your_final_prediction.csv')
