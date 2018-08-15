from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
import tensorflow.contrib.losses as tf_losses
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from nets import *
def plot_hist(pred_prob, actl_lbl):
    pred_prob_0 = pred_prob[actl_lbl==0]
    pred_prob_1 = pred_prob[actl_lbl==1]
    
    bins = np.linspace(0, 1, 100)
    
    plt.hist(pred_prob_0, bins, alpha=0.5, label='label 0')
    plt.hist(pred_prob_1, bins, alpha=0.5, label='label 1')
    plt.legend(loc='upper right')
    plt.savefig('123.png')
    plt.close()
class My_research_tuning():
    def __init__(self, network, feature_size):

        self.X = tf.placeholder(tf.float32, shape=[None, feature_size])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')
        
        self.pred = network(self.X)
        self.diff = tf.reduce_mean(self.pred - self.y)
        self.total_loss = tf.nn.l2_loss(self.pred - self.y)
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.total_loss, var_list=network.vars)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        


    def feature_tuning(self):

        self.sess.run(tf.global_variables_initializer())
        data_train_real = np.loadtxt('feature_train_real.csv',dtype=float,delimiter=',')
        data_train_syntheic = np.loadtxt('feature_train_syntheic.csv',dtype=float,delimiter=',')
        
        X_train_real = data_train_real[:,:-1]
        y_train_real = data_train_real[:,-1]
        X_train_syntheic = data_train_syntheic[:,:-1]
        y_train_syntheic = data_train_syntheic[:,-1]

        feature_size = X_train_real.shape[1]

        batch_size = 32
        syntheic_size = X_train_syntheic.shape[0]
        real_size = X_train_real.shape[0]

        # train model
        for epoch in range(50000):
            X_train_real_batch = X_train_real[np.random.randint(real_size, size=batch_size), :]
            y_batch = np.zeros((batch_size, 1))
            mask = np.random.randint(feature_size, size=batch_size)
            idx1 = 0
            for idx2 in mask:
                y_batch[idx1][0] = X_train_real_batch[idx1][idx2]
                X_train_real_batch[idx1][idx2] = 0
                idx1 += 1
            # print(X_train_real_batch)
            _ , loss, error, y_pred = self.sess.run([self.train_step, self.total_loss, self.diff, self.pred],
                    feed_dict={self.X: X_train_real_batch, self.y: y_batch , self.is_training: True})
            if epoch % 5000 == 0:
                print(loss)

        #tuning feature
        X_tuning_start = np.array(X_train_syntheic)
        for epoch in range(20):
            idx1 = 0
            for batch_count in range(syntheic_size//batch_size+1):
                X_tuning_batch = X_tuning_start[batch_count*batch_size:(batch_count+1)*batch_size]
                # print(X_tuning_batch.shape)
                for idx2 in range(feature_size):
                    feature_tmp = np.array(X_tuning_batch)
                    feature_tmp[:, idx2] = 0
                    # print(feature_tmp)
                    X_tuned = self.sess.run(self.pred, feed_dict={self.X: feature_tmp, self.is_training: False})
                    # print(y_tuned)
                    # print(idx1, idx2)
                    if idx2 == 0:
                        X_tuned_total = X_tuned
                    else:
                        X_tuned_total = np.column_stack((X_tuned_total, X_tuned))
                # print(y_tuned_total)

                    
                if idx1 == 0 :
                    X_tuning = X_tuned_total
                else:
                    X_tuning = np.vstack((X_tuning, X_tuned_total))
                idx1 += 1
            # print(X_tuning)
            X_tuning_start = np.array(X_tuning)
        X_tuned = X_tuning_start
        X_tuned = np.column_stack((X_tuned, np.ones((syntheic_size,1))))
        np.savetxt('feature_train_syntheic_tuned.csv', X_tuned, delimiter=',') 


def final_predict():
    data_train_real = np.loadtxt('feature_train_real.csv',dtype=float,delimiter=',')
    data_train_syntheic = np.loadtxt('feature_train_syntheic.csv',dtype=float,delimiter=',')
    # data_train_syntheic = np.loadtxt('feature_train_syntheic_tuned.csv',dtype=float,delimiter=',')

    data_test = np.loadtxt('feature_test.csv',dtype=float,delimiter=',')

    # X_train = data_train_real[:,:-1]
    # y_train = data_train_real[:,-1]
    data_train = np.append(data_train_syntheic, data_train_real, axis=0)
    # data_train = data_train_real
    
    X_train = data_train[:,:-1]
    y_train = data_train[:,-1]
    # y_train = np.append(data_train_syntheic[:,:-1], data_train_real[:,-1], axis=0)
    print(X_train.shape)
    print(y_train.shape)

    # data_original = np.loadtxt('./datas/Ecoil.csv',dtype=float,delimiter=',')
    # data_train, data_test = train_test_split(data_original, test_size=0.3, random_state=None)
    # X_train = data_train[:,:-1]
    # y_train = data_train[:,-1]

    # clf = SVC(kernel='poly', probability=True)
    # clf = SVC(kernel='linear', probability=True)
    # clf = RandomForestClassifier(max_depth=5, random_state=2)
    clf = LogisticRegression()
    # clf = SVC(kernel='rbf', probability=True)


    #clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train) 
    print('training done')
    X_test = data_test[1:,:-1]
    y_test = data_test[1:,-1]

    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    # y_pred_prob_train = clf.predict_proba(X_train)

    # print(y_test)
    # print(y_pred_prob[:,1])
    # print(y_pred)

    confusion_matrix_test = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_test)
    print(classification_report(y_test,y_pred))
    AUC = roc_auc_score(y_test,y_pred_prob[:, 1])

    print(AUC)
    plot_hist(y_pred_prob[:, 1], y_test)

    return AUC

    # acc, pre, rec, F1 = eval(confusion_matrix_test)
    # print('Accurancy: {:.4}; precision: {:.4}; recall: {:.4}; F1-score: {:.4}'.format(acc, pre, rec, F1))
