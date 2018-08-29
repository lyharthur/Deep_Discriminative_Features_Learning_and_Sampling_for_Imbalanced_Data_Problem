import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
import numpy as np
import matplotlib as mpl
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import os,sys
import random
from PIL import Image
import scipy.misc
from sklearn.svm import SVC
import tensorflow.contrib.losses as tf_losses

from nets import *
from datas import *
from tfrecord import *
from triplet_loss import *
from t_sne import *

def squared_dist(A): 
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    
    return distances

def get_center_loss(features, labels, alpha, beta, gamma, num_classes, ratio):
    
    centers = tf.get_variable('centers', [num_classes, features.get_shape()[1]], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    labels = tf.cast(labels, tf.int32)
    # print(centers,labels)
    centers_batch = tf.gather(centers, labels)

    weights_batch = tf.cast(tf.gather(ratio, labels), tf.float32)

    distance_batch = tf.reduce_sum(features - centers_batch, 1)
    weighted_distance_batch = tf.multiply(distance_batch, weights_batch)
    # weighted_distance_batch = tf.divide(distance_batch, weights_batch)
    weighted_loss = tf.nn.l2_loss(weighted_distance_batch)

    # var_target_batch = tf.cast(tf.pow(tf.gather(ratio, labels),2), tf.float32)*gamma

    # output = sum(t ** 2) / 2
    centers_loss = tf.nn.l2_loss(features - centers_batch)


    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers = tf.scatter_sub(centers, labels, diff)

    pairwise_centers_distance = tf.reduce_mean(squared_dist(centers))
    pairwise_centers_distance_loss = tf.pow(tf.maximum(beta - pairwise_centers_distance, 0), 2)

    # varaince loss
    # centers_distance_batch = tf.pow((features - centers_batch),2)
    # varaince_batch = tf.reduce_mean(tf.divide(centers_distance_batch, tf.cast(appear_times, tf.float32)), 1)

    # varaince_loss = tf.reduce_sum(tf.divide(tf.abs(varaince_batch - var_target_batch), weights_batch))

    # original center loss + imbalance ratio weight
    # loss = weighted_loss

    # original center loss 
    # loss = weighted_loss + varaince_loss + pairwise_centers_distance_loss

    loss = centers_loss + pairwise_centers_distance_loss

    t1 = centers_loss
    t2 = centers_loss
    t3 = centers_loss 
    return loss, centers, t1, t2, t3

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
 
def sample_y(m, n): # 16 , class_num , fig count
    y = np.zeros([m,n])
    for i in range(m):
        y[i, i%n] = 1
    #y[:,7] = 1
    #y[-1,0] = 1
    #print(y)
    return y

def concat(z,y):
    return tf.concat([z,y],1)
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

def evaluation(confusion_matrix):
    acc = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[3,3] + confusion_matrix[4,4])/np.sum(confusion_matrix)

    sensitivity_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0] + confusion_matrix[2,0] + confusion_matrix[3,0] + confusion_matrix[4,0]) 
    sensitivity_1 = confusion_matrix[1,1] / (confusion_matrix[0,1] + confusion_matrix[1,1] + confusion_matrix[2,1] + confusion_matrix[3,1] + confusion_matrix[4,1]) 
    sensitivity_2 = confusion_matrix[2,2] / (confusion_matrix[0,2] + confusion_matrix[1,2] + confusion_matrix[2,2] + confusion_matrix[3,2] + confusion_matrix[4,2]) 
    sensitivity_3 = confusion_matrix[3,3] / (confusion_matrix[0,3] + confusion_matrix[1,3] + confusion_matrix[2,3] + confusion_matrix[3,3] + confusion_matrix[4,3]) 
    sensitivity_4 = confusion_matrix[4,4] / (confusion_matrix[0,4] + confusion_matrix[1,4] + confusion_matrix[2,4] + confusion_matrix[3,4] + confusion_matrix[4,4])

    specificity_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[0,2] + confusion_matrix[0,3] + confusion_matrix[0,4]) 
    specificity_1 = confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1] + confusion_matrix[1,2] + confusion_matrix[1,3] + confusion_matrix[1,4]) 
    specificity_2 = confusion_matrix[2,2] / (confusion_matrix[2,0] + confusion_matrix[2,1] + confusion_matrix[2,2] + confusion_matrix[2,3] + confusion_matrix[2,4]) 
    specificity_3 = confusion_matrix[3,3] / (confusion_matrix[3,0] + confusion_matrix[3,1] + confusion_matrix[3,2] + confusion_matrix[3,3] + confusion_matrix[3,4]) 
    specificity_4 = confusion_matrix[4,4] / (confusion_matrix[4,0] + confusion_matrix[4,1] + confusion_matrix[4,2] + confusion_matrix[4,3] + confusion_matrix[4,4]) 

    print('ACC:',acc)
    print('sensitivity:')
    print(sensitivity_0,sensitivity_1,sensitivity_2,sensitivity_3,sensitivity_4)
    print('specificity:')
    print(specificity_0,specificity_1,specificity_2,specificity_3,specificity_4)

    return 


class Classifer():
    def  __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data
        # self.min = [0,1,2,3,4,5,6,7]
        # self.min = [0,1,2,3,4]
        # self.min = [0]        
        self.min = [1]

        self.class_num = self.data.class_num # condition
        self.size = self.data.size
        self.channel = self.data.channel
        self.feature_size = 32

        self.ratio = tf.placeholder(tf.int32, shape=[self.class_num])
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.class_num])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        self.A_feature, self.logits = self.classifier(self.X, self.feature_size, self.is_training)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        argmax_prediction = tf.argmax(self.logits, 1)
        argmax_y = tf.argmax(self.y, 1)

        # self.TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        # self.TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        # self.FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        # self.FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
        # self.precision = self.TP / (self.TP + self.FP)
        # self.recall = self.TP / (self.TP + self.FN)

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(argmax_y, argmax_prediction, num_classes=self.class_num)
        #self.auc = tf.metrics.auc(argmax_y, argmax_prediction)

        ratio = tf.cast(self.ratio, tf.float32)
        weights = tf.reduce_sum(ratio * self.y, axis=1)

        self.cross_entropy = tf.reduce_mean(tf.multiply(weights, tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y)))
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cross_entropy,  var_list=self.classifier.vars)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 8200, batch_size = 256, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print(self.data.filename_train,self.data.filename_val)
        image, label = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        #image, label = read_and_decode('train_slm_512_01234.tfrecords',batch_size,self.size)
        #image_val, label_val = read_and_decode('test_slm_512_01234.tfrecords',batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)
        
        ratio=[0,0,0,0,0,0,0,0,0,0]
        # ratio=[0,0]
        for _ in range(self.class_num):
            if _ in self.min:
                ratio[_] = self.data.imb_ratio
            else: 
                ratio[_] = 1

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_b, y_1, y_b = get_batch(self.sess, image, label, self.class_num, batch_size, one_hot=False)

                self.sess.run(
                    [self.train_step],
                    feed_dict={self.X: X_b, self.y: y_b, self.ratio: ratio, self.is_training: True})
            
            # save img, model. print loss
            if epoch % 500 == 0 or epoch <= 10:

                #C_loss_curr, C_acc_curr, TP, TN, FP, FN, precision, recall, confusion_matrix = self.sess.run(
                feature, C_loss_curr, C_acc_curr, confusion_matrix = self.sess.run(	
                        #[self.cross_entropy, self.accuracy, self.TP, self.TN, self.FP, self.FN, self.precision, self.recall,self.confusion_matrix],
                        [self.A_feature, self.cross_entropy, self.accuracy, self.confusion_matrix],
                        feed_dict={self.X: X_b, self.y: y_b, self.ratio: ratio, self.is_training: False})
                #print(epoch, C_loss_curr, C_acc_curr)
                #print('Iter: {}; C_loss: {:.4}; C_acc: {:.4}  TP: {}; TN: {:}; FP: {:}; FN: {:}; precision: {:}; recall: {:}'.format(epoch, C_loss_curr, C_acc_curr,TP,TN,FP,FN,precision,recall))
                print('Iter: {}; C_loss: {:.4}; C_acc: {:.4};'.format(epoch, C_loss_curr, C_acc_curr))
                
                # print("Computing t-SNE embedding")           
                # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                # X_tsne = tsne.fit_transform(feature)

                # plot_embedding(X_tsne, y_1, "t-SNE embedding") 
                #print(''.format(TP,TN,FP,FN,precision,recall))
                # print(confusion_matrix)
                #evaluation(confusion_matrix)
                #acc, sens, spec = eval(confusion_matrix)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

                # validation
                X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.class_num, batch_size)
                y_pred, confusion_matrix_final = self.sess.run(
                            [self.logits, self.confusion_matrix],
                            feed_dict={self.X: X_val_b, self.y: y_val_b, self.ratio: ratio, self.is_training: False})
                y_val_b_total = np.argmax(y_val_b, axis=1)
                y_pred_total = np.argmax(y_pred, axis=1)

                if epoch % 1000 == 0 :
                    batch_number = self.data.len_val//batch_size
                    #print(batch_number)
                    for _ in range(batch_number):
                        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.class_num, batch_size)
                        y_pred, confusion_matrix_val = self.sess.run(
                                        [self.logits, self.confusion_matrix],
                                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.ratio: ratio, self.is_training: False})
                        y_val_b = np.argmax(y_val_b, axis=1)
                        y_pred = np.argmax(y_pred, axis=1)

                        y_val_b_total = np.append(y_val_b_total, y_val_b)
                        y_pred_total = np.append(y_pred_total, y_pred)

                        # print(y_val_b_total.shape, y_pred_total.shape)
                        confusion_matrix_final += confusion_matrix_val
                    print(confusion_matrix_final)
                    print(classification_report(y_val_b_total, y_pred_total))
                    #print(Acc_test)
                    # acc, pre, rec, F1 = eval(confusion_matrix_final)
                    # print('Accurancy: {:.4}; precision: {:.4}; recall: {:.4}; F1-score: {:.4}'.format(acc, pre, rec, F1))
                #evaluation(confusion_matrix_final)
                #print('AUC {:.4}'.format(AUC_val))
                #acc, sens, spec = eval(confusion_matrix_final)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

    def test(self, data_val, batch_size = 20):
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        batch_number = self.data_val.len//batch_size
        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.class_num, batch_size)
        confusion_matrix_final = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.ratio: ratio, self.is_training: False})
        for _ in range(batch_number-1):
            X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.class_num, batch_size)
            confusion_matrix_val = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.ratio: ratio, self.is_training: False})

            confusion_matrix_final+=confusion_matrix_val
        print(confusion_matrix_final)
        evaluation(confusion_matrix)

        #acc, sens, spec = eval(confusion_matrix_final)
        #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")            


class My_research():
    def  __init__(self, net, data):
        self.network = net
        self.data = data
        # self.min = [0,1,2,3,4,5,6,7]
        # self.min = [0,1,2,3,4]
        self.min = [1]
        #self.class_num = self.data.class_num # num of classes
        self.class_num = 10 # 10 -> 2
        self.size = self.data.size
        # self.feature_size = 64 #mnist
        self.feature_size = 32 #cifar
        self.channel = self.data.channel

        # 建立一个变量，存储每一类的中心，不训练
        # self.centers = tf.get_variable('centers', [self.class_num, self.feature_size], dtype=tf.float32,
        #     initializer=tf.constant_initializer(0), trainable=False)
                
        self.ratio = tf.placeholder(tf.int32, shape=[self.class_num])
        self.X_A = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y_A = tf.placeholder(tf.int32, shape=[None, 1])
        self.y_A_one = tf.placeholder(tf.float32, shape=[None, self.class_num])

        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        #self.y = tf.equal(tf.argmax(self.y_1), tf.argmax(self.y_2))

        _lamda = 0.001
        #0~1
        _alpha = 0.5
        _beta = 100
        _gamma = 1
        self.margin = 0.5


        self.A_feature, self.A_fc = self.network(self.X_A, self.feature_size, self.is_training)

        
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.A_fc, 1), tf.argmax(self.y_A_one, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        ratio = tf.cast(self.ratio, tf.float32)
        weights = tf.reduce_sum(ratio * self.y_A_one, axis=1)

        self.cross_entropy = tf.reduce_mean(tf.multiply(weights, tf.nn.softmax_cross_entropy_with_logits(logits = self.A_fc, labels = self.y_A_one)))

        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.A_fc, labels = self.y_A_one))
        
        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.y_A_one, 1), tf.argmax(self.A_fc, 1), num_classes=self.class_num)

        # self.triplet_loss, self.hardest_positive_dist, self.hardest_negative_dist = batch_hard_triplet_loss(self.y_A, self.A_feature, self.margin)

        self.center_loss, self.centers, self.t1, self.t2, self.t3 = get_center_loss(self.A_feature, self.y_A, _alpha, _beta, _gamma, self.data.class_num, self.ratio)

        self.total_loss = self.cross_entropy + self.center_loss
        # self.total_loss = self.triplet_loss

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.total_loss,  var_list=self.network.vars)
        # self.train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(self.total_loss,  var_list=self.triplet.vars)
        # self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4,decay=0.9,momentum=0.1).minimize(self.total_loss,  var_list=self.triplet.vars)
        #self.train_step = tf.train.MomentumOptimizer(1e-4, 0.7, use_nesterov=True).minimize(self.total_loss,  var_list=self.one_shot.vars)

        self.saver = tf.train.Saver(max_to_keep=3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 1200, batch_size = 256, restore = True):
         
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        print(self.data.filename_train)
        image1, label1 = read_and_decode(self.data.filename_train, batch_size,self.size)
        # image2, label2 = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        ratio=[0,0,0,0,0,0,0,0,0,0]
        # ratio=[0,0]
        for _ in range(self.class_num):
            if _ in self.min:
                ratio[_] = self.data.imb_ratio
            else: 
                ratio[_] = 1

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_1, y_1, y_1_one = get_batch(self.sess, image1, label1, self.class_num, batch_size, one_hot=False)

                _ = self.sess.run(
                        [self.train_step],
                        feed_dict={self.X_A: X_1, self.y_A: y_1, self.y_A_one: y_1_one, self.ratio: ratio, self.is_training: True})
                #print(accuracy_R, accuracy_L)
                #print(test_out4)

            # save img, model. print loss
            if epoch % 200 == 0 or epoch <= 50:
                t1, t2, t3, centor_loss, feature, total_loss, acc= self.sess.run(  
                # C_loss_curr, distance, accuracy_L, accuracy_R = self.sess.run(  
                        [self.t1, self.t2, self.t3, self.center_loss, self.A_feature, self.total_loss, self.accuracy],
                        feed_dict={self.X_A: X_1, self.y_A: y_1, self.y_A_one: y_1_one, self.ratio: ratio, self.is_training: False})


                print('Iter: {}; C_loss: {:.4};'.format(epoch, total_loss))
                # print(t1, t2, t3)
                print('ACC', acc)
                # print('centor_loss',centor_loss)
                # print('PN',P[:1],N[:1])
                # print('triplet:', triplet_L)#, 'autoencoder', MSE_L)
                

                if epoch % 500 == 0: 
                    #plot
                    # print("Computing t-SNE embedding")           
                    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                    # X_tsne = tsne.fit_transform(feature)

                    # plot_embedding(X_tsne, y_1, "t-SNE")      

                    # print("PCA")
                    # pca = PCA(n_components=2)
                    # X_pca = pca.fit_transform(feature)
                    # plot_embedding(X_pca, y_1, "pca")  

                    # validation
                    X_val_b, y_val_b, y_val_one = get_batch(self.sess, image_val, label_val, self.class_num, batch_size, one_hot=False)
                    y_pred, confusion_matrix_final = self.sess.run(
                                [self.A_fc, self.confusion_matrix],
                                feed_dict={self.X_A: X_val_b, self.y_A: y_val_b, self.y_A_one: y_val_one, self.ratio: ratio, self.is_training: False})
                    y_val_b_total = np.argmax(y_val_one, axis=1)
                    y_pred_total = np.argmax(y_pred, axis=1)

                    batch_number = self.data.len_val//batch_size
                    #print(batch_number)
                    for _ in range(batch_number):
                        X_val_b, y_val_b, y_val_one = get_batch(self.sess, image_val, label_val, self.class_num, batch_size, one_hot=False)
                        y_pred, confusion_matrix_val = self.sess.run(
                                        [self.A_fc, self.confusion_matrix],
                                        feed_dict={self.X_A: X_val_b, self.y_A: y_val_b, self.y_A_one: y_val_one, self.ratio: ratio, self.is_training: False})
                        y_val_one = np.argmax(y_val_one, axis=1)
                        y_pred = np.argmax(y_pred, axis=1)

                        y_val_b_total = np.append(y_val_b_total, y_val_one)
                        y_pred_total = np.append(y_pred_total, y_pred)

                        # print(y_val_b_total.shape, y_pred_total.shape)
                        confusion_matrix_final += confusion_matrix_val
                    print(confusion_matrix_final)
                    print(classification_report(y_val_b_total, y_pred_total))
                    

    def get_feature(self, batch_size = 50):
        image_train, label_train = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        #batch_number = self.data_val.len//batch_size
        feature_space_num = self.feature_size
        feature_space_out = [np.zeros(feature_space_num+1)]
        feature_space_out = np.array(feature_space_out)
        for _ in range((self.data.len_train_Maj+self.data.len_train_min)//batch_size + 1):
            X, y, y_one= get_batch(self.sess, image_train, label_train, self.class_num, batch_size, one_hot=False)
            feature_space = self.sess.run(self.A_feature, feed_dict={self.X_A: X, self.is_training: False})
            #print(y_val_b[:5], feature[:5])
            
            feature_space = np.append(feature_space, y, axis=1)
            feature_space_out = np.append(feature_space_out, feature_space, axis=0)
            #print(feature_space_out[:3])

        #print(feature_space_out)
        #print(y_out)
        #np.append(feature_space_out, y_out, axis=1)
        # sperate 0,1
        feature_space_min = []
        feature_space_Maj = []
        for i in range(feature_space_out.shape[0]-1):
            # print(feature_space_out[i+1, -1])

            if feature_space_out[i+1, -1] in self.min:
                feature_space_min.append(feature_space_out[i+1,:])
            else:
                feature_space_Maj.append(feature_space_out[i+1,:])
            # if feature_space_out[i+1, -1]==1:

        feature_space_Maj = np.array(feature_space_Maj)
        feature_space_min = np.array(feature_space_min)
        #print(feature_space_min.shape[0])
        feature_mean_Maj = np.mean(feature_space_Maj, axis=0)
        feature_mean_min = np.mean(feature_space_min, axis=0)
        #print(feature_mean_Maj[:10])
        #print(feature_mean_min[:10])
        #print(feature_space_min[:3])
        #print(np.subtract(feature_space_min, feature_mean_Maj))
        feature_mean_Maj = np.array(feature_mean_Maj[:-1])
        feature_mean_min = np.array(feature_mean_min[:-1])

        feature_space_min_new = []

        X_fake = np.zeros((1, self.size))
        # create temporary feature
        count_syntheic = 0
        while(count_syntheic < (self.data.len_train_min)):
        # while(count_syntheic < (self.data.count_t1)//2):
        #for _ in range(1):
            tmp = []
            for i in range(feature_space_num+1):
                r = random.randint(0,feature_space_min.shape[0]-1)
                tmp.append(feature_space_min[r, i])

            tmp = np.array(tmp)
            # tmp_input = tmp[:-1]
            # tmp_input = tmp_input.reshape(-1,feature_space_num)
            
            # print(tmp_input,tmp_input.shape)
            # print(feature_mean_Maj, feature_mean_min)

            feature_mean_Maj_min = (feature_mean_Maj + feature_mean_min)/2
            # print(tmp)
            # print(feature_mean_Maj_min)

            dist = np.linalg.norm(feature_mean_Maj-feature_mean_min)
            dist_Maj_min = np.linalg.norm(feature_mean_Maj_min-tmp[:-1])
            dist_Maj = np.linalg.norm(feature_mean_Maj-tmp[:-1])
            dist_min = np.linalg.norm(feature_mean_min-tmp[:-1])

            # print(dist, dist_Maj, dist_Maj_min, dist_min)
            # pred= self.sess.run(self.pred, feed_dict={self.X: X, self.feature_tmp: tmp_input, self.verify: True})
            # print(pred)

            #dist/2 < dist_Maj-dist_min and 
            # if dist_Maj>dist_Maj_min and dist_Maj_min>dist_min:
            feature_space_min_new.append(tmp)
            count_syntheic +=1
        feature_space_min_new = np.array(feature_space_min_new)

        print(feature_space_min_new.shape)
        print(feature_space_Maj.shape)
        print(feature_space_min.shape)

        
        np.savetxt('feature_train_syntheic.csv', feature_space_min_new, delimiter=',') 
        feature_space_all = np.append(feature_space_Maj, feature_space_min, axis=0)
        np.savetxt('feature_train_real.csv', feature_space_all, delimiter=',')

        #clf = SVC()
        #clf.fit(X,y)
        print('get test')
        # get test
        feature_space_test = [np.zeros(feature_space_num+1)]
        feature_space_test = np.array(feature_space_test)
        print(self.data.len_val)
        for _ in range(self.data.len_val//batch_size+1):
            X_val, y_val, y_val_one = get_batch(self.sess, image_val, label_val, self.class_num, batch_size, one_hot=False)
            feature_space = self.sess.run(self.A_feature, feed_dict={self.X_A: X_val, self.is_training: False})
            feature_space = np.append(feature_space, y_val, axis=1)
            feature_space_test = np.append(feature_space_test, feature_space, axis=0)
            #print(_)
        X_test = feature_space_test[1:,:-1]
        y_test = feature_space_test[1:,-1]
        #print(X_test[:5])
        #print(y_test[:5])
        np.savetxt('feature_test.csv', feature_space_test, delimiter=',') 

    def test(self, data):
        return 0


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")          

class My_research_csv():
    def  __init__(self, net, data):
        self.network = net
        # self.min = [0,1,2,3,4]
        # self.min = [0]
        self.class_num = data.class_num # num of classes
        self.size = data.size

        self.feature_size = 8

        
        self.centers_tensor = tf.placeholder(dtype=tf.float32, shape=[self.class_num, self.feature_size])
                
        self.ratio = tf.placeholder(tf.int32, shape=[self.class_num], name='ratio')

        self.X = tf.placeholder(tf.float32, shape=[None, self.size], name='X')
        # self.feature_tmp = tf.placeholder(tf.float32, shape=[None, self.feature_size], name='feature_tmp')
        self.y = tf.placeholder(tf.float32, shape=[None, self.class_num], name='y')
        self.learning_rate = tf.placeholder(tf.float32)

        # self.verify = tf.placeholder(tf.bool, name='verify')

        #self.y = tf.equal(tf.argmax(self.y_1), tf.argmax(self.y_2))

        _lamda = 0.1
        #0~1
        _alpha = 0.5
        _beta = 100
        _gamma = 1
        self.margin = 0.2

        # self.pred, self.feature = self.network(self.X, self.feature_size)
        self.mse, self.feature = self.network(self.X, self.feature_size, self.size)
        
        # self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1)), tf.float32)
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        ratio = tf.cast(self.ratio, tf.float32)
        weights = tf.reduce_sum(ratio * self.y, axis=1)


        # self.cross_entropy = tf.reduce_mean(tf.multiply(weights, tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y)))

        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y))
        
        # self.confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.y, 1), tf.argmax(self.pred, 1), num_classes=self.class_num)

        self.triplet_loss, self.hardest_positive_dist, self.hardest_negative_dist = batch_hard_triplet_loss(tf.argmax(self.y, 1), self.feature, self.margin)

        # self.center_loss, self.centers, self.t1, self.t2, self.t3 = get_center_loss(self.feature, tf.argmax(self.y, 1), _alpha, _beta, _gamma, self.class_num, self.ratio)

        # self.total_loss = self.cross_entropy + self.triplet_loss
        # self.total_loss = self.cross_entropy + self.center_loss*_lamda
        # self.total_loss = self.mse 
        self.total_loss = self.triplet_loss*_lamda + self.mse
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss, var_list=self.network.vars)
        # self.train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(self.total_loss,  var_list=self.network.vars)
        # self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4,decay=0.9,momentum=0.1).minimize(self.total_loss,  var_list=self.network.vars)
        #self.train_step = tf.train.MomentumOptimizer(1e-4, 0.7, use_nesterov=True).minimize(self.total_loss,  var_list=self.network.vars)

        self.saver = tf.train.Saver(max_to_keep=3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, data, ckpt_dir, training_epoches = 8500, batch_size = 32, restore = True):
        self.data = data
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        

        # centers_value = np.zeros([self.class_num, self.feature_size], dtype=np.float32)
        # print(self.data.filename_train)
        # print(self.size)
        self.attr_train, self.label_train = self.data.read_and_decode_csv_tfrecord(self.data.filename_train, batch_size, self.size)
        self.atrr_val, self.label_val = self.data.read_and_decode_csv_tfrecord(self.data.filename_val, batch_size, self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        print('ratio Train',self.data.count_t0, self.data.count_t1)
        print('ratio Val',self.data.count_v0, self.data.count_v1)
        ratio=[self.data.count_t1, self.data.count_t0]
        learning_rate = 1e-3

        for epoch in range(training_epoches):
            # update C
        
            # real label to train C
            X_train, y_train = self.data.get_batch_tfrecord(self.sess, self.attr_train, self.label_train, self.class_num, batch_size)
            # X_train, y_train = self.data.get_batch(self.data.train_set, batch_size, self.data.size)
            #print(X_train)
            # print(X_train, y_train)
            # _, _c = self.sess.run([self.train_step, self.centers],
            _ = self.sess.run(self.train_step,
                    feed_dict={self.X: X_train, self.y: y_train, self.ratio: ratio, self.learning_rate: learning_rate})


            # save img, model. print loss
            if epoch % 1000 == 0 or epoch <= 10:
                feature, total_loss= self.sess.run(  
                        [self.feature, self.total_loss],
                        feed_dict={self.X: X_train, self.y: y_train, self.ratio: ratio})


                print('Iter: {}; C_loss: {:.4};'.format(epoch, total_loss))
                # print(t1, t2, t3)
                # print(feature)
                # print('ACC', acc)
                # print('PN',P[:1],N[:1])
                # print('triplet:', triplet_L)#, 'autoencoder', MSE_L)


                # if epoch % 1000 == 0 and epoch > 1000: 
                    # learning_rate *=0.5
                #     #plot t-SNE
                #     # print("Computing t-SNE embedding")           
                #     # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                #     # X_tsne = tsne.fit_transform(feature)
                #     # plot_embedding(X_tsne, y_train, "t-SNE")   

                    print("PCA")
                    # pca = PCA(n_components=2)
                    # X_pca = pca.fit_transform(feature)
                    # plot_embedding(X_pca, y_train, "pca") 

                    

    def get_feature(self, test, batch_size = 64):
        self.attr_train, self.label_train = self.data.read_and_decode_csv_tfrecord(self.data.filename_train, batch_size, self.size)
        self.attr_val, self.label_val = self.data.read_and_decode_csv_tfrecord(self.data.filename_val, batch_size, self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        #batch_number = self.data_val.len//batch_size
        feature_space_num = self.feature_size
        feature_space_out = [np.zeros(feature_space_num+1)]
        feature_space_out = np.array(feature_space_out)

        

        self.data.batch_count = 0
        for _ in range((self.data.count_t)//batch_size+1):
            # X, y = self.data.get_batch(self.data.train_set, batch_size, self.data.size)
            X, y= self.data.get_batch_tfrecord(self.sess, self.attr_train, self.label_train, self.class_num, batch_size)
            # print(y)
            y = np.argmax(y, axis=1).reshape((-1, 1))
            feature_space = self.sess.run(self.feature, feed_dict={self.X: X})
            #print(y_val_b[:5], feature[:5])
            # print(y)
            # print(feature_space.shape,y.shape)

            feature_space = np.append(feature_space, y, axis=1)
            # print(feature_space.shape,feature_space_out.shape)
            feature_space_out = np.append(feature_space_out, feature_space, axis=0)
            #print(feature_space_out[:3])

        #print(feature_space_out)
        #print(y_out)
        #np.append(feature_space_out, y_out, axis=1)
        # sperate 0,1
        feature_space_min = []
        feature_space_Maj = []
        for i in range(feature_space_out.shape[0]-1):
            # print(feature_space_out[i+1, -1])

            if feature_space_out[i+1, -1] == 1:
                feature_space_min.append(feature_space_out[i+1,:])
            else:
                feature_space_Maj.append(feature_space_out[i+1,:])
            # if feature_space_out[i+1, -1]==1:

        feature_space_Maj = np.array(feature_space_Maj)
        feature_space_min = np.array(feature_space_min)
        #print(feature_space_min.shape[0])
        feature_mean_Maj = np.mean(feature_space_Maj, axis=0)
        feature_mean_min = np.mean(feature_space_min, axis=0)
        #print(feature_mean_Maj[:10])
        #print(feature_mean_min[:10])
        #print(feature_space_min[:3])
        #print(np.subtract(feature_space_min, feature_mean_Maj))
        feature_mean_Maj = np.array(feature_mean_Maj[:-1])
        feature_mean_min = np.array(feature_mean_min[:-1])
        feature_space_min_new = []
        
        X_fake = np.zeros((1, self.size))
        # create temporary feature
        count_syntheic = 0
        while(count_syntheic < (self.data.count_t0 - self.data.count_t1)//2):
        # while(count_syntheic < (self.data.count_t1)*2):
        #for _ in range(1):
            tmp = []
            for i in range(feature_space_num+1):
                r = random.randint(0,feature_space_min.shape[0]-1)
                tmp.append(feature_space_min[r, i])

            tmp = np.array(tmp)
            # tmp_input = tmp[:-1]
            # tmp_input = tmp_input.reshape(-1,feature_space_num)
            
            # print(tmp_input,tmp_input.shape)
            # print(feature_mean_Maj, feature_mean_min)
            feature_mean_Maj_min = (feature_mean_Maj + feature_mean_min)/2

            dist = np.linalg.norm(feature_mean_Maj-feature_mean_min)
            dist_Maj_min = np.linalg.norm(feature_mean_Maj_min-tmp[:-1])
            dist_Maj = np.linalg.norm(feature_mean_Maj-tmp[:-1])
            dist_min = np.linalg.norm(feature_mean_min-tmp[:-1])

            # print(dist, dist_Maj, dist_Maj_min, dist_min)
            # pred= self.sess.run(self.pred, feed_dict={self.X: X, self.feature_tmp: tmp_input, self.verify: True})
            # print(pred)

            # dist/2 < dist_Maj-dist_min and
            if dist_Maj>dist_Maj_min and dist_Maj_min>dist_min:
                feature_space_min_new.append(tmp)
                count_syntheic +=1
        feature_space_min_new = np.array(feature_space_min_new)

        print(feature_space_min_new.shape)
        print(feature_space_Maj.shape)
        print(feature_space_min.shape)

        np.savetxt('feature_train_syntheic.csv', feature_space_min_new, delimiter=',')

        # feature_space_all = np.append(feature_space_min_new, feature_space_Maj, axis=0)
        # feature_space_all = np.append(feature_space_all, feature_space_min, axis=0)

        feature_space_all = np.append(feature_space_Maj, feature_space_min, axis=0)
        # print(feature_space_all)

        np.savetxt('feature_train_real.csv', feature_space_all, delimiter=',') 
        # X = feature_space_all[:,:-1]
        # y = feature_space_all[:,-1]
        

        print('get test')
        # get test
        feature_space_test = [np.zeros(feature_space_num+1)]
        feature_space_test = np.array(feature_space_test)
        print(self.data.count_v)


        self.data.batch_count = 0
        for _ in range(self.data.count_v//batch_size+1):
            # X_val, y_val = self.data.get_batch(self.data.test_set, batch_size, self.data.size)
            X_val, y_val = self.data.get_batch_tfrecord(self.sess, self.attr_val, self.label_val, self.class_num, batch_size)

            y_val = np.argmax(y_val, axis=1).reshape((-1, 1))
            feature_space = self.sess.run(self.feature, feed_dict={self.X: X_val})
            # print(X_val, feature_space,y_val)
            feature_space = np.append(feature_space, y_val, axis=1)
            feature_space_test = np.append(feature_space_test, feature_space, axis=0)
            #print(_)
        X_test = feature_space_test[1:,:-1]
        y_test = feature_space_test[1:,-1]
        #print(X_test[:5])
        #print(y_test[:5])
        np.savetxt('feature_test.csv', feature_space_test, delimiter=',') 


    def test(self, data):
        return 0


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")          
