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
print('\nmean column-wise ROC AUC on Train data: ', np.mean(train_rocs))
print('mean column-wise ROC AUC on Val data:', np.mean(valid_rocs))


