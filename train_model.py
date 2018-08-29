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
from models import *
from triplet_loss import *
from tuning_and_eval import *

import tensorflow.contrib.slim.nets as nets

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    #train_data_folder = './datas/imagenet/train/'
    #val_data_folder = './datas/imagenet/test/'

    #train_data_folder = './datas/tsmc/train/'
    #val_data_folder = './datas/tsmc/test/'

    #'dog,cat,bird,fish,insect,food,plant,rabbit,scenery,snake'

    # train_data_folder = './datas/dog_cat/train/'
    # val_data_folder = './datas/dog_cat/test/'
    
    # train_data_folder = './datas/mnist/train-images/'
    # val_data_folder = './datas/mnist/test-images/'   

    # train_data_folder = './datas/fashion/train-images/'
    # test_data_folder = './datas/fashion/test-images/'

    # train_data_folder = './datas/cifar10/train-images/'
    # val_data_folder = './datas/cifar10/test-images/'

    # train_data_folder = './datas/test_data.csv'

    # train_data_folder = './datas/diabete.csv'
    train_data_folder = './datas/Haberman.csv'
    # train_data_folder = './datas/satimage.csv'
    # train_data_folder = './datas/Ecoil.csv'
    # train_data_folder = './datas/shuttle.csv'
    # train_data_folder = './datas/ionosphere.csv'
    # train_data_folder = './datas/vehicle.csv'
    # train_data_folder = './datas/give_me_some_credit.csv'

    folder = './'
    model = sys.argv[1]
    img_size = sys.argv[2]
    class_id = sys.argv[3]
    class_num = sys.argv[4]
    imb_ratio = sys.argv[5]
    restore = sys.argv[6]
    print('Model: '+model +'; Img_Resize: '+img_size +'; class_ID: '+class_id +'; class_num: '+class_num +'; imb_ratio: '+imb_ratio +'; Restore_ckpt: '+restore)
    
    
    if model == 'cnn' :

        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_class_id+'/'

        # param
        #classifier = C_conv(size=int(img_size),class_num=int(class_num))
        classifier = cnn_mnist(class_num=int(class_num))
        # classifier = cnn_cifar(class_num=int(class_num))

        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), imb_ratio=int(imb_ratio))
        #data = mydata(data_folder=test_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        
        #data_val = mydata(data_folder=val_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        C = Classifer(classifier, data)
        if restore == 'True':
            C.restore_ckpt(ckpt_folder)
            C.train(ckpt_dir=ckpt_folder, restore=True, batch_size=128)
        else:
            C.train(ckpt_dir=ckpt_folder, restore=False, batch_size=128)

    elif model == 'my' :
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/triplet_'+img_size+'_'+new_class_id+'/'
        #restore_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+class_id+'/'

        # network 
        network = cnn_mnist(class_num=int(class_num))

         
        data = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), imb_ratio=int(imb_ratio))

        # run
        model = My_research(network, data)
        if restore == 'True':
            model.restore_ckpt(ckpt_folder)
            #GAN.restore_ckpt(folder+'ckpt/W_GAN_64_13/')
            #GAN.discriminator = discriminator
            #GAN.sess.run(tf.variables_initializer(GAN.discriminator.vars))
            model.train(ckpt_dir=ckpt_folder, restore=True, batch_size=128)
            model.get_feature()
        else:
            model.train(ckpt_dir=ckpt_folder, restore=False, batch_size=128)
            model.get_feature()
            AUC = final_predict()
            print(AUC)

    elif model == 'my_csv' :
        AUC_total = 0
        data = mydata_csv(data_file=train_data_folder, class_num=int(class_num))

        # network 
        network_0 = my_csv_AE(class_num=int(class_num))
        # network_0 = my_csv(class_num=int(class_num))

        #build
        model_1 = My_research_csv(network_0, data)

        # run
        for _ in range(5):
            ckpt_folder = folder+'ckpt/mycsv_'+img_size+'/'
            data = mydata_csv(data_file=train_data_folder, class_num=int(class_num))

            model_1.train(data, ckpt_dir=ckpt_folder, restore=False, batch_size=64)
            for _in in range(5):
                model_1.get_feature(str(_in), 16)
                # model_2.feature_tuning()
                AUC = final_predict()
                AUC_total += AUC
        print(AUC_total/25)
        
    elif model == 'test' :
        AUC_total = 0
        for _ in range(100):
            AUC = final_predict()
            AUC_total += AUC
        print(AUC_total/100)
    else:
        print('Wrong model')
