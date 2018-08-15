import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

from nets import *
from datas import *
from gans import *

import tensorflow.contrib.slim.nets as nets

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    #train_data_folder = './datas/imagenet/train/'
    #val_data_folder = './datas/imagenet/test/'

    #train_data_folder = './datas/tsmc/train/'
    #val_data_folder = './datas/tsmc/test/'

    #'dog,cat,bird,fish,insect,food,plant,rabbit,scenery,snake'

    #train_data_folder = './datas/dog_cat/train/'
    #val_data_folder = './datas/dog_cat/test/'
    
    #train_data_folder = './datas/mnist/train-images/'
    #val_data_folder = './datas/mnist/test-images/'   

    train_data_folder = './datas/fashion/train-images/'
    test_data_folder = './datas/fashion/test-images/'

    #train_data_folder = './data/cifar10/train-images/'
    #val_data_folder = './data/cifar10/test-images/'
    folder = './'
    model = sys.argv[1]
    img_size = sys.argv[2]
    class_id = sys.argv[3]
    class_num = sys.argv[4]
    sample_num = sys.argv[5]

    
    print('Model: '+model +'; Img_Resize: '+img_size +'; class_id: '+class_id +'; class_Num: '+class_num +'; Sample_num: '+sample_num)
    
    if model == 'wgan':
  	
        sample_folder = folder+'Samples_single/DR_'+img_size+'_'+class_id+'_Wgan_conv'
        ckpt_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        restore_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=1)
        
        # run
        GAN = WGAN(generator, discriminator, data)
        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))

    elif model == 'gan_c' :#not finish

        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        sample_folder = folder+'Samples/DR_'+img_size+'_'+new_class_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'
        restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv_condition(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(class_num))
         
        data = mydata(size=int(img_size), defect=class_id, class_num=int(class_num))

        # run
        GAN = GAN_Classifier(generator, discriminator, classifier, data)

        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))
    elif model == 'began':
    
        sample_folder = folder+'Samples_single/DR_'+img_size+'_'+class_id+'_'+'began_conv'
        ckpt_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+class_id+'/'
        restore_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+class_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        generator = G_conv_BEGAN(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=class_id, class_num=1)
        
        # run
        GAN = BEGAN(generator, discriminator, data, flag=False)
        GAN.restore_ckpt(ckpt_folder)
        GAN.test(sample_folder,int(sample_num))
    elif model == 'c':
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/'+'classifier_'+img_size+'_'+new_class_id+'/'     

        classifier = my_inception_v3(class_num=int(class_num))
        #classifier = net_in_net(class_num=int(class_num))

        #data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        data_val = mydata(data_folder=val_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        C = Classifer(classifier, data_val, data_val)
        C.restore_ckpt(ckpt_folder)
        C.test(data_val)

    elif model == 'ae':
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/'+'ae_'+img_size+'_'+new_class_id+'/'     

        autoencoder = VAutoencoder(int(img_size),128)
        #classifier = net_in_net(class_num=int(class_num))

        #data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        A = Autoencoder(autoencoder, data)
        A.restore_ckpt(ckpt_folder)
        A.test(data,int(sample_num))
    elif model == 'one_shot':
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/'+'one_shot_'+img_size+'_'+new_class_id+'/'     

        one_shot = one_shot()

        #data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        data = mydata(data_folder=test_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        One_Shot_Learning = one_shot_learning(one_shot, data)

        One_Shot_Learning.restore_ckpt(ckpt_folder)
        One_Shot_Learning.test()

    else: print('wrong model')

