from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from sklearn.metrics import classification_report, roc_auc_score

def eval(confusion_matrix):
    TN = confusion_matrix[0,0] 
    FN = confusion_matrix[0,1] 
    FP = confusion_matrix[1,0] 
    TP = confusion_matrix[1,1] 
    precision = TP / (TP+FN)
    recall = TP / (TP+FP)
    acc = (TP+TN) / (TP+FN+TN+FP)
    F1 = 2*precision*recall/(recall+precision)
    return acc, precision, recall, F1


data_train = np.loadtxt('feature_one_shot_train.csv',dtype=float,delimiter=',')
data_test = np.loadtxt('feature_one_shot_test.csv',dtype=float,delimiter=',')


X_train = data_train[:,:-1]
y_train = data_train[:,-1]
#print(X_train)
# print(y_train)
clf = SVC(kernel='poly')
# clf = SVC(kernel='linear')
# clf = SVC(kernel='rbf')


#clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train) 
print('training done')
X_test = data_test[1:,:-1]
y_test = data_test[1:,-1]

y_pred = clf.predict(X_test)
confusion_matrix_test = confusion_matrix(y_test, y_pred)
print(confusion_matrix_test)
print(classification_report(y_test,y_pred))
# print(roc_auc_score(y_test,y_pred))

# acc, pre, rec, F1 = eval(confusion_matrix_test)
# print('Accurancy: {:.4}; precision: {:.4}; recall: {:.4}; F1-score: {:.4}'.format(acc, pre, rec, F1))
