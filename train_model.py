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

    train_data_folder = './datas/fashion/train-images/'
    test_data_folder = './datas/fashion/test-images/'

    # train_data_folder = './datas/cifar10/train-images/'
    # val_data_folder = './datas/cifar10/test-images/'

    # train_data_folder = './datas/test_data.csv'

    # train_data_folder = './datas/diabete.csv'
    # train_data_folder = './datas/Haberman.csv'
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
    
    if model == 'gan_c' :
        
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        sample_folder = folder+'Samples/CW_GAN_'+img_size+'_'+new_class_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'
        #restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(class_num))
        #classifier = nielsen_net(class_num=int(class_num))
        #classifier = net_in_net(class_num=int(class_num))
        #classifier = my_inception_v3(class_num=int(class_num))

        min_class_id = class_id.split(',')[0]
        print('min:',min_class_id)
        data_all = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        data_min = mydata(data_folder=train_data_folder, size=int(img_size), classes=min_class_id, class_num=1)
        data_val = mydata(data_folder=val_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)

        # run
        GAN = GAN_Classifier(generator, discriminator, classifier, data_all, data_min, data_val)
        if restore == 'True':
            GAN.restore_ckpt(ckpt_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size=64, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size=64, restore=False)

#######
    elif model == 'wgan':
        sample_folder = folder+'Samples/DR_'+img_size+'_'+class_id+'_wgan_conv'
        ckpt_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        #restore_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        
        # run
        GAN = WGAN(generator, discriminator, data)
        if restore == 'True':
            GAN.restore_ckpt(ckpt_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 64, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 64, restore=False)
        
    elif model == 'c' :

        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_class_id+'/'

        # param
        #classifier = nielsen_net(class_num=int(class_num))
        #classifier = net_in_net(class_num=int(class_num))
        classifier = my_inception_v3(class_num=int(class_num))
        #classifier = C_conv(size=int(img_size),class_num=int(class_num))
         
        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        #data_val = mydata(data_folder=val_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        C = Classifer(classifier, data)
        if restore == 'True':
            C.restore_ckpt(ckpt_folder)
            C.train(ckpt_dir=ckpt_folder, restore=True, batch_size=128)
        else:
            C.train(ckpt_dir=ckpt_folder, restore=False, batch_size=128)
    elif model == 'cnn' :

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


    elif model == 'ae' :

        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/ae_'+img_size+'_'+new_class_id+'/'

        # param
        #classifier = nielsen_net(class_num=int(class_num))
        #classifier = net_in_net(class_num=int(class_num))
        autoencoder = one_shot_cifar()
        #classifier = C_conv(size=int(img_size),class_num=int(class_num))
         
        data = mydata(data_folder=train_data_folder, size=int(img_size), classes=class_id, class_num=int(class_num))
        # run
        A = Autoencoder(autoencoder, data)
        if restore == 'True':
            A.restore_ckpt(ckpt_folder)
            A.train(ckpt_dir=ckpt_folder, restore=True, batch_size=128)
        else:
            A.train(ckpt_dir=ckpt_folder, restore=False, batch_size=128)

    elif model == 'one_shot' :
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/one_shot_'+img_size+'_'+new_class_id+'/'
        #restore_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+class_id+'/'

        # param
        #one_shot = one_shot()
        one_shot = one_shot_cifar(class_num=int(class_num))
        # one_shot = one_shot_mnist()
         
        data = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), imb_ratio=int(imb_ratio))
        #data_2 = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num))

        #data_val = mydata(data_folder=val_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        One_Shot_Learning = one_shot_learning(one_shot, data)
        if restore == 'True':
            One_Shot_Learning.restore_ckpt(ckpt_folder)
            #GAN.restore_ckpt(folder+'ckpt/W_GAN_64_13/')
            #GAN.discriminator = discriminator
            #GAN.sess.run(tf.variables_initializer(GAN.discriminator.vars))
            One_Shot_Learning.train(ckpt_dir=ckpt_folder, restore=True, batch_size=256)
            One_Shot_Learning.get_feature()
        else:
            One_Shot_Learning.train(ckpt_dir=ckpt_folder, restore=False, batch_size=256)
            One_Shot_Learning.get_feature()

    elif model == 'triplet' :
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
        # triplet = one_shot()
        triplet = one_shot_cifar(class_num=int(class_num))
        # triplet = one_shot_mnist(class_num=int(class_num))
         
        data = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), imb_ratio=int(imb_ratio))
        #data_2 = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num))

        #data_val = mydata(data_folder=val_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
        # run
        Triplet = triplet_learning(triplet, data)
        if restore == 'True':
            Triplet.restore_ckpt(ckpt_folder)
            #GAN.restore_ckpt(folder+'ckpt/W_GAN_64_13/')
            #GAN.discriminator = discriminator
            #GAN.sess.run(tf.variables_initializer(GAN.discriminator.vars))
            Triplet.train(ckpt_dir=ckpt_folder, restore=True, batch_size=128)
            Triplet.get_feature()
        else:
            Triplet.train(ckpt_dir=ckpt_folder, restore=False, batch_size=128)
            Triplet.get_feature()

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
        # network = cnn_cifar(class_num=int(class_num))
        network = cnn_mnist(class_num=int(class_num))

         
        data = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), imb_ratio=int(imb_ratio))
        #data_2 = mydata(data_folder=train_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num))

        #data_val = mydata(data_folder=val_data_folder,size=int(img_size), classes=class_id, class_num=int(class_num), is_val=True)
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
        # network_1 = my_csv(class_num=int(class_num))
        # network_2 = my_tuning()

        #build
        model_1 = My_research_csv(network_0, data)
        # model_1 = My_research_csv(network_1, data)
        # model_2 = My_research_tuning(network_2, model_1.feature_size)

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
