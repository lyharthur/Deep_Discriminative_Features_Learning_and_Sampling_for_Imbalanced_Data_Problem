import os,sys
import tensorflow as tf
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

from tfrecord import *

#data_folder = '/home/project/itriDR/pre_data/train_slm_512/'
def get_img(img_path, resize_h):
    img=scipy.misc.imread(img_path).astype(np.float) # mode L for grayscale
    #print(img.shape)
     
    #print(img.shape)
    #crop resize  Original Use
    resize_h = resize_h
    resize_w = resize_h
    #h, w = img.shape[:2]
    #j = int(round((h - crop_h)/2.))
    #i = int(round((w - crop_w)/2.))
    #cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])# cropp
    img = scipy.misc.imresize(img,[resize_h, resize_w])# no cropp
    #print(cropped_image.shape)
    #img = cropped_image#for class grayscale data 
    #img = np.dstack((cropped_image,cropped_image))[:,:,:1]
    #img = cropped_image.reshape((resize_h,resize_w,3))
    #print(img)
    #print(img.shape)
    return np.array(img)/255.0

class mydata_csv():
    def __init__(self, data_file, class_num, is_val=False):
        datas = np.loadtxt(data_file, dtype=float,delimiter=',')

        self.class_num = class_num
        self.size = datas.shape[1]-1
        print('#atrr:', self.size)
        data_name = data_file.split('/')[2][:-4]
        self.filename_train = './tfrecords/' + data_name + '_train.tfrecords'
        self.filename_val = './tfrecords/' + data_name + '_val.tfrecords'

        writer_train = tf.python_io.TFRecordWriter(self.filename_train)
        writer_val = tf.python_io.TFRecordWriter(self.filename_val)

        self.count_t = 0
        self.count_v = 0
        self.count_t0 = 0
        self.count_t1 = 0
        self.count_v0 = 0
        self.count_v1 = 0
        self.batch_count = 0
        # self.train_set = np.zeros((1, self.size+1))
        # self.test_set = np.zeros((1, self.size+1))

        self.train_set, self.test_set = train_test_split(datas, test_size=0.3, random_state=None)
        for data in self.train_set:
            # r = random.randint(1,10)
            # print(data)
            x = data[:-1]
            y = data[-1]

            # x_mean = np.mean(x, axis = 0)
            # x_std = np.std(x, axis = 0)

            # x -= x_mean # zero-center
            # x /= x_std # normalize

            if y == 0 :
                down = random.randint(1,5)
                if down == 1:
                    self.count_t0 += 1
            if y == 1 :
                self.count_t1 += 1
                down = 1

            x = x.astype(np.float32)
            x = x.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
                    'atrr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
                })) 
            if down ==1:
                writer_train.write(example.SerializeToString())
                self.count_t+=1
            # data = np.reshape(data, (1, self.size+1))
            # self.train_set = np.append(self.train_set, data, axis=0)     

        for data in self.test_set:
            # r = random.randint(1,10)
            # print(data)
            x = data[:-1]
            y = data[-1]

            # x -= x_mean # zero-center
            # x /= x_std # normalize

            if y == 0 :
                down = random.randint(1,1)
                if down == 1:
                    self.count_v0 += 1
            if y == 1 :
                self.count_v1 += 1
                down = 1

            x = x.astype(np.float32)
            x = x.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
                    'atrr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
                })) 
            if down ==1:
                writer_val.write(example.SerializeToString())
                self.count_v+=1
            # data = np.reshape(data, (1, self.size+1))
            # self.test_set = np.append(self.test_set, data, axis=0) 


        # self.train_set = self.train_set[1:]
        # self.test_set = self.test_set[1:]
        print('Train, Test', self.count_t, self.count_v)
        # print(self.train_set)
        # print(self.test_set)
        # data_train = np.array(data_train)
        # data_test = np.array(data_test)
        # print(data_train.shape, data_test.shape)

        # X_train = data_train[:,:-1]
        # y_train = data_train[:,-1]
        
        # X_test = data_test[1:,:-1]
        # y_test = data_test[1:,-1]

    def get_batch(self, dataset, batch_size, attr_size):
        # print(self.batch_count)
        batch_number = len(dataset)/batch_size

        X = dataset[:, :-1]
        y = dataset[:, -1]
        # y = y.reshape(len(dataset),1)

        X_b = X[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        y_b = y[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        # patch = batch_size - len(y_b)
        # if patch > 0:
        #     X_b = X[self.batch_count*batch_size-patch:(self.batch_count+1)*batch_size]
        #     y_b = y[self.batch_count*batch_size-patch:(self.batch_count+1)*batch_size]

        one_hot = np.zeros((len(y_b),self.class_num))
        for i,val in enumerate(y_b):
            one_hot[i,int(val)]=1
        y_b_onehot = one_hot

        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0
            np.random.shuffle(dataset)
        return X_b, y_b_onehot

    def read_and_decode_csv_tfrecord(self, filename, batch_size, attr_size): 
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.float32),
                                               'atrr' : tf.FixedLenFeature([], tf.string),
                                           })

        label = tf.cast(features['label'], tf.int32)
        atrr = tf.decode_raw(features['atrr'], tf.float32)
        atrr = tf.reshape(atrr, [attr_size])
        
        atrr_b, label_b = tf.train.shuffle_batch([atrr, label],
                                       batch_size=batch_size, capacity=100+3*batch_size,num_threads=64,
                                       min_after_dequeue=100)

        return atrr_b, label_b

    def get_batch_tfrecord(self, sess, atrr, label, classes_num, batch_size):
        X_b, y_b = sess.run([atrr,label])
        #print(y_b.shape)
        
        #one_hot
        one_hot = np.zeros((batch_size,classes_num))###self.y_dim  # my im gan = 2
        for i,val in enumerate(y_b):
            # print(i,val)
            one_hot[i,val]=1
        label_onehot = one_hot
        #print(X_b.shape,label_onehot.shape)

        return X_b, label_onehot

class mydata():
    def __init__(self, data_folder, size, classes, class_num, imb_ratio, is_val=False):

        self.z_dim = 512
        self.size = size
        self.class_num = class_num
        self.imb_ratio = imb_ratio
        self.channel = 3 ##
        
        #old version?
        data_list = []
        new_id = ''
        for i in range(class_num):
            class_sub = classes.split(',')[i]
            new_id += class_sub
            #print(class_sub)
            datapath = data_folder+class_sub+'/*' #HERE
            print(datapath)
            data_list.extend(glob(datapath))
        print('data_num:',len(data_list))

        self.filename_train = './tfrecords/'+ data_folder.split('/')[2] + '-' + data_folder.split('/')[3]+'-'+new_id+'_train.tfrecords'
        self.filename_val = './tfrecords/'+ data_folder.split('/')[2] + '-' + data_folder.split('/')[3]+'-'+new_id+'_val.tfrecords'

        print(self.filename_train)
        #TFRecord
        self.len_train_Maj, self.len_train_min, self.len_val = create_TFR(classes,class_num, imb_ratio=imb_ratio, #
                                    filename_train=self.filename_train,filename_val=self.filename_val, #
                                    folder=data_folder,img_size=self.size,is_val=is_val)

        print('Maj, min, val')
        print(self.len_train_Maj, self.len_train_min, self.len_val)
        '''
        #get_img
        img_list = [get_img(img_path, self.size) for img_path in data_list]
        self.data = img_list

        #datapath = data_folder+'data/'+class+'/'
        
    
        #self.data = glob(os.path.join(datapath, '*.jpg'))
        #print(self.data)

        # old version?
        label = [] 
        check = []
        label_count = -1
        for path in data_list: 
            class_id = path.split('/')[6]#7,5
            #print(class_id)
            if class_id not in check:
                check.append(class_id)
                label_count+=1
            label.append(label_count)

        #print(label)
        one_hot = np.zeros((len(label),self.y_dim))###self.y_dim  # my im gan = 2
        for i,val in enumerate(label):
            one_hot[i,val]=1
        #print(one_hot)
        self.label = one_hot
        #print(len(label)) 
        self.batch_count = 0

        tmp = list(zip(self.data, self.label))
        random.shuffle(tmp)
        self.data ,self.label = zip(*tmp)'''

    def __call__(self,batch_size):
        #print(img_b)
        #old version
        batch_number = len(self.data)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0

            tmp = list(zip(self.data, self.label))
            random.shuffle(tmp)
            self.data ,self.label = zip(*tmp)

        img_b = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        label_b = self.label[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

        #batch = [get_img(img_path, self.size*3, self.size) for img_path in path_list]
        img_b = np.array(img_b).astype(np.float32)
        
       

        '''
        print self.batch_count
        fig = self.data2fig(batch_imgs[:16,:,:])
        plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        '''
        #print(len(label_list))
        return img_b, label_b

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
   
        for i, sample in enumerate(samples):
            #print(sample.shape)
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #print(sample.shape)
            #new_sample = np.concatenate((sample,sample),axis = 2) ## for 1d to 3d
            #new_sample = np.concatenate((new_sample,sample),axis = 2) ## for 1d to 3d
            #print(new_sample.shape)
            #sample = new_sample ## for 1d to 3d
            plt.imshow(sample)
        return fig

class mnist():
    def __init__(self, flag='conv', is_tanh = False):
        datapath = folder+'GAN_yhliu/MNIST_data'
        self.X_dim = 784 # for mlp
        self.z_dim = 100
        self.y_dim = 10
        self.size = 28 # for conv
        self.channel = 1 # for conv
        self.data = input_data.read_data_sets(datapath, one_hot=True)
        self.flag = flag
        self.is_tanh = is_tanh

    def __call__(self,batch_size):
        batch_imgs,y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
        if self.is_tanh:
            batch_imgs = batch_imgs*2 - 1        
        return batch_imgs, y

    def data2fig(self, samples):
        if self.is_tanh:
            samples = (samples + 1)/2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
        return fig    

if __name__ == '__main__':
    data = face3D()
    print(data(17).shape)
