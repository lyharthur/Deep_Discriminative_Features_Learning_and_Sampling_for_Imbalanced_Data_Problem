from sklearn.svm import SVC

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from t_sne import *
from sklearn.model_selection import train_test_split
from sklearn import manifold

data_train_real = np.loadtxt('feature_train_real.csv',dtype=float,delimiter=',')
data_train_syntheic = np.loadtxt('feature_train_syntheic.csv',dtype=float,delimiter=',')
data_train_syntheic[:,-1] = 2
# data_train_syntheic = np.loadtxt('feature_train_syntheic_tuned.csv',dtype=float,delimiter=',')

data_test = np.loadtxt('feature_test.csv',dtype=float,delimiter=',')

data = np.append(data_train_syntheic, data_train_real, axis=0)
data = np.append(data, data_test, axis=0)

# data, data_test = train_test_split(data, test_size=0.8, random_state=None)

feature = data[:,:-1]
y_train = data[:,-1]

print(y_train)
print("PCA")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
pca = PCA(n_components=2, svd_solver='arpack')

X_pca = pca.fit_transform(feature)
# print(X_pca)
plot_embedding(X_pca, y_train, "pca_") 


data_csv = np.loadtxt('./datas/diabete.csv',dtype=float,delimiter=',')
X = data_csv[:,:-1]
y = data_csv[:,-1]

SM = SMOTE(ratio='minority',kind='regular')
X_res, y_res = SM.fit_sample(X, y)
y_res[len(y):] = 2

print("PCA_smote")

X_pca = pca.fit_transform(X_res)
# print(X_pca)
plot_embedding(X_pca, y_res, "PCA_smote") 



X = data_csv[:,:-1]
y = data_csv[:,-1]

adasyn = ADASYN(ratio='minority')
X_res, y_res = adasyn.fit_sample(X, y)
y_res[len(y):] = 2

print("PCA_ADASYN")

X_pca = pca.fit_transform(X_res)
# print(X_pca)
plot_embedding(X_pca, y_res, "PCA_ADASYN") 



X = data_csv[:,:-1]
y = data_csv[:,-1]

sampler = RandomOverSampler(ratio='minority')
X_res, y_res = sampler.fit_sample(X, y)

y_res[len(y):] = 2

print("PCA_RandomOverSampler")

X_pca = pca.fit_transform(X_res)

plot_embedding(X_pca, y_res, "PCA_RO") 
