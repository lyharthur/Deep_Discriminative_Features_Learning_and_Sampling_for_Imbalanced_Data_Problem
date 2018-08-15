import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np
import random
from glob import glob

def random_pre_process(images): 
    #random_flip_up_down
    # images = tf.image.random_flip_up_down(images) 
    #random_flip_left_right
    images = tf.image.random_flip_left_right(images) 
    #random_brightness
    images = tf.image.random_brightness(images, max_delta=0.3) 
    #random_contrast
    images = tf.image.random_contrast(images, 0.9, 1.1)
    #random_saturation
    tf.image.random_saturation(images, 0.4, 0.5)
    #new_size = tf.constant([image_size,image_size],dtype=tf.int32)
    #images = tf.image.resize_images(images, new_size)
    return images

def create_TFR(classes, class_num, imb_ratio, filename_train, filename_val, folder, img_size, is_val):
    data_list = []
    new_id = ''
    for i in range(class_num):
        class_sub = classes.split(',')[i]
        new_id += class_sub
        #print(class_sub)
        datapath = folder+class_sub+'/*' #HERE
        #print(datapath)
        data_list.extend(glob(datapath))

    #print(folder)
    #print(classes)
    label = [] 
    check = []
    label_count = -1
    for path in data_list: 
        #print(path)
        class_id = path.split('/')[4]#DR 6  #4 MNIST
        #print(class_id)
        if class_id not in check:
            check.append(class_id)
            label_count+=1
        label.append(label_count)

    writer_train = tf.python_io.TFRecordWriter(filename_train)
    writer_val = tf.python_io.TFRecordWriter(filename_val)

    tmp = list(zip(data_list, label))
    random.shuffle(tmp)
    
    count_t_Maj = 0
    count_t_min = 0
    count_v = 0
    for img_path,index in tmp:
        #print(img_path)
        #down_sampling
        if is_val == False:
            r=1
            # if index == 0 or index == 1 or index == 2 or index == 3 or index == 4 or index == 5 or index == 6 or index == 7 : #min
            # if index == 0 or index == 1 or index == 2 or index == 3 or index == 4 :
            if index == 1:
                r = random.randint(1,imb_ratio) # 3/5
                # index = 0
                #if r==1 or r==2 or r==3: r=1
            #if index == 1 :or index == 6 or index == 7 or index == 8 or index == 9:
            else: #Maj
                r = 1
                # index = 1


            if r==1:
                
                img=Image.open(img_path)
                img= img.resize((img_size,img_size))
                if img.mode == 'L':
                    #print('0')
                    img = np.array(img) 
                    img = np.stack((img,)*3,axis=-1) # for grayscale to 3-channel 
                    #print(img.shape)
                else:
                    img = np.array(img)
                    #print(img.shape)
                
                img_raw=img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) 
                train_val = random.randint(1,10)
                #val  8:2
                if train_val == 1 or train_val == 2 :
                    writer_val.write(example.SerializeToString()) 
                    count_v+=1
                #train
                else:
                    #down sampleing 
                    # if index == 0 or index == 1 or index == 2 or index == 3 or index == 4 or index == 5 or index == 6 or index == 7:
                    # if index == 0 or index == 1 or index == 2 or index == 3 or index == 4 :
                    if index == 1 :

                        # oversampling
                        # for _ in range(25):
                        #     writer_train.write(example.SerializeToString())
                        #     count_t_min+=1
                        writer_train.write(example.SerializeToString())
                        count_t_min+=1
                        # count_t_Maj+=1
                    else :
                        #down = random.randint(1,10) #downsampling
                        #if down == 1 :

                        # oversampling
                        # for _ in range(4):
                        #      writer_train.write(example.SerializeToString())
                        #      count_t_min+=1
                        writer_train.write(example.SerializeToString())
                        count_t_Maj+=1
                        # count_t_min+=1



    writer_train.close()
    writer_val.close()
    return count_t_Maj, count_t_min, count_v
def read_and_decode(filename,batch_size,img_size): 
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 3])  
    img = random_pre_process(img)
    img = tf.cast(img, tf.float32) * (1. / 255) 
    #print(img)
    label = tf.cast(features['label'], tf.int32) 
    
    #one_hot = np.zeros((len(label),5))###self.y_dim  # my im gan = 2
    #for i,val in enumerate(label):
    #    one_hot[i,val]=1
    #label_onehot = one_hot

    img_b, label_b = tf.train.shuffle_batch([img, label],
                                   batch_size=batch_size, capacity=10000+3*batch_size,num_threads=64,
                                   min_after_dequeue=10000)

    return img_b, label_b


def get_batch(sess,image,label,classes_num,batch_size,one_hot=True):
    #img_out, label_out = sess.run([image,label])
    X_b, y_b = sess.run([image,label])
    #print(y_b.shape)
    if one_hot:
        #one_hot
        one_hot = np.zeros((batch_size,classes_num))###self.y_dim  # my im gan = 2
        for i,val in enumerate(y_b):
            one_hot[i,val]=1
        label_onehot = one_hot
        #print(X_b.shape,label_onehot.shape)

        return X_b, label_onehot

    else:
        #normal
        label_out = np.zeros((batch_size,1))
        for i,val in enumerate(y_b):
            label_out[i,0]=val

        one_hot = np.zeros((batch_size,classes_num))###self.y_dim  # my im gan = 2
        for i,val in enumerate(y_b):
            one_hot[i,val]=1
        label_onehot = one_hot
        #print(X_b.shape,label_onehot.shape)

        return X_b, label_out, label_onehot
