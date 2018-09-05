import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.slim as slim

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    This function has been taken from link-> https://github.com/tensorflow/tensorflow/issues/2169
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

class my_tuning(object):
    def __init__(self):
        self.name = 'my_tuning'

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.fully_connected(inputs, 16, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='fc1')
            # net = tcl.fully_connected(net, 16, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='fc2')
            net = tcl.fully_connected(net, 16, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='feature')
            y_pred = tcl.fully_connected(net, 1, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return y_pred
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class my_csv_AE(object):
    def __init__(self, class_num):
        self.name = 'my_csv_AE'
        self.class_num = class_num
        
    def __call__(self, inputs, feature_size, data_size, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #net = tcl.fully_connected(inputs, feature_size//4, activation_fn=lrelu, normalizer_fn=None, scope='fc1')
            encoder = tcl.fully_connected(inputs, feature_size//2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='e')
            feature = tcl.fully_connected(encoder, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature')
            decoder = tcl.fully_connected(feature, feature_size//2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='d')
            out = tcl.fully_connected(decoder, data_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='out')
            # y_pred = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')
            mse = tf.nn.l2_loss(inputs - out)

            return mse, feature
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class my_csv(object):
    def __init__(self, class_num):
        self.name = 'my_csv'
        self.class_num = class_num

    def __call__(self, inputs, feature_size, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #net = tcl.fully_connected(inputs, feature_size//4, activation_fn=lrelu, normalizer_fn=None, scope='fc1')
            net1 = tcl.fully_connected(inputs, feature_size//2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='fc2')      
            feature = tcl.fully_connected(net1, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature')
            net2 = tcl.fully_connected(feature, feature_size//2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm, scope='fc3')
            y_pred = tcl.fully_connected(net2, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return y_pred, feature
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class my_inception_v3(object):
    def __init__(self, class_num):
        self.name = 'InceptionV3'
        self.class_num = class_num

    def __call__(self, inputs, is_training, reuse=False,dropout_keep_prob=0.3):
        
        y_pred, end_ps = slim.nets.inception.inception_v3(inputs=inputs, num_classes=self.class_num, is_training=is_training, reuse=reuse)
        
        return y_pred
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
class cnn(object):
    def __init__(self, class_num):
        self.name = 'cnn'
        self.class_num = 10

    def __call__(self, inputs, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=4, kernel_size=3,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1')
            net = tcl.dropout(net, keep_prob=0.8)
            net = tcl.max_pool2d(net,kernel_size=2, padding='SAME')

            net = tcl.conv2d(net, num_outputs=8, kernel_size=3,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            net = tcl.dropout(net, keep_prob=0.5)
            net = tcl.max_pool2d(net,kernel_size=2, padding='SAME')

            net = tcl.conv2d(net, num_outputs=16, kernel_size=3,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv3')
            net = tcl.dropout(net, keep_prob=0.5)
            print(net.shape)
            net = tcl.flatten(net) #feature
            print(net.shape)
            feature = tcl.fully_connected(net, 512, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='feature')

            y_pred = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return y_pred
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class cnn_mnist(object):
    def __init__(self, class_num):
        self.name = 'cnn'
        self.class_num = 10

    def __call__(self, inputs, feature_size, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=4, kernel_size=3,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1')
            print(net.shape)
            net = tcl.conv2d(net, num_outputs=8, kernel_size=3,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            print(net.shape)

            #net = tcl.dropout(net, keep_prob=0.5)  
            net = tcl.flatten(net) #feature
            print(net.shape)
            feature = tcl.fully_connected(net, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature')
            y_pred = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return feature, y_pred
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class cnn_cifar(object):
    def __init__(self, class_num):
        self.name = 'cnn'
        self.class_num = class_num

    def __call__(self, inputs, feature_size, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=64, kernel_size=5,# 
                        stride=1, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1') 
            net = tcl.max_pool2d(net, kernel_size=3, stride=2)
            print(net.shape)
            net = tcl.conv2d(net, num_outputs=64, kernel_size=5,# 
                        stride=1, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            net = tcl.max_pool2d(net, kernel_size=3, stride=2)
            print(net.shape)

            net = tcl.flatten(net) #feature
            print(net.shape)
            net = tcl.fully_connected(net, 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='fc1')
            net = tcl.fully_connected(net, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='fc2')
            feature = tcl.fully_connected(net, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature', weights_initializer=tf.random_normal_initializer(0, 0.02))

            y_pred = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='out')

            return feature, y_pred
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class nielsen_net(object):
    def __init__(self, class_num):
        self.name = 'nielsen_net'
        self.class_num = class_num

    def __call__(self, inputs, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
            net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

            # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
            net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')
            print(net.shape)
            # Reshape: 5x5x40 => 1000x1
            net = tf.reshape(net, [-1, 30*30*40]) #about img size

            # Fully Connected Layer: 1000x1 => 1000x1
            net = slim.fully_connected(net, 1000, scope='layer5')
            net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

            # Second Fully Connected: 1000x1 => 1000x1
            net = slim.fully_connected(net, 1000, scope='layer6')
            net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

            # Output Layer: 1000x1 => 10x1
            net = slim.fully_connected(net, self.class_num, scope='output')# 2 class
            net = slim.dropout(net, is_training=is_training, scope='output-dropout')

            return net
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class one_shot_cifar(object):
    def __init__(self, class_num):
        self.name = 'one_shot_cifar'
        self.class_num = class_num

    def __call__(self, inputs, feature_size, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=16, kernel_size=5,# 
                        stride=1, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1') 
            net = tcl.max_pool2d(net, kernel_size=3, stride=2)
            print(net.shape)
            net = tcl.conv2d(net, num_outputs=32, kernel_size=5,# 
                        stride=1, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            net = tcl.max_pool2d(net, kernel_size=3, stride=2)
            print(net.shape)
            net = tcl.conv2d(net, num_outputs=64, kernel_size=5,# 
                        stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv3')
            # net = tcl.max_pool2d(net, kernel_size=2, stride=2)
            print(net.shape)
            # net = tcl.conv2d(net, num_outputs=64, kernel_size=3,# 
            #             stride=2, activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv4')
            # print(net.shape)
            
            net = tcl.flatten(net) #feature
            net = tcl.fully_connected(net, 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='fc1', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(net.shape)
            feature = tcl.fully_connected(net, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature', weights_initializer=tf.random_normal_initializer(0, 0.02))


            fc = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            # return feature, mse, x_out, fc
            return feature, fc
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class one_shot_mnist(object):
    def __init__(self, class_num):
        self.name = 'one_shot_mnist'
        self.class_num = class_num       

    def __call__(self, inputs, feature_size, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=4, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1') 
            print(net.shape)
            net = tcl.conv2d(net, num_outputs=8, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            print(net.shape)

             
            net = tcl.flatten(net) #feature
            print(net.shape)
            feature = tcl.fully_connected(net, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature')

            decoder = tcl.fully_connected(feature, 392, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope='fc')
            print(decoder.shape)
            decoder = tf.reshape(decoder, (-1, 7, 7, 8))

            #decoder = tcl.conv2d_transpose(decoder, 16, 3, stride=2, # size*4
            #                        activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 4, 3, stride=2, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(decoder.shape)
            x_out = tcl.conv2d_transpose(decoder, 1, 3, stride=2, # size*16
                                    activation_fn=lrelu, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(x_out.shape)
            mse = tf.reduce_mean(tf.pow(255.*(inputs - x_out),2))
            # #mse = tf.reduce_mean(tf.reduce_sum((inputs - x_out)**2, 1))
            # mse = tf.reduce_mean(tf.abs(inputs - x_out))
            
            fc = tcl.fully_connected(feature, self.class_num, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return feature, mse, x_out, fc#
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class one_shot(object):
    def __init__(self):
        self.name = 'one_shot'

    def __call__(self, inputs, feature_size, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = tcl.conv2d(inputs, num_outputs=4, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv1')
            
            net = tcl.conv2d(net, num_outputs=8, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv2')
            #net = tcl.dropout(net, keep_prob=0.5)
            
            net = tcl.conv2d(net, num_outputs=16, kernel_size=3,# 
                        stride=1, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv3')
            print(net.shape)

            net = tcl.conv2d(net, num_outputs=32, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv4')
            print(net.shape)

            net = tcl.conv2d(net, num_outputs=32, kernel_size=3,# 
                        stride=1, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv5')
            print(net.shape)

            net = tcl.conv2d(net, num_outputs=64, kernel_size=3,# 
                        stride=2, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv6')
            print(net.shape)

            net = tcl.conv2d(net, num_outputs=64, kernel_size=3,# 
                        stride=1, activation_fn=lrelu, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv7')
            print(net.shape)
            #net = tcl.dropout(net, keep_prob=0.5)
            
            #net = tcl.conv2d(net, num_outputs=32, kernel_size=3,# 
            #            stride=2, activation_fn=None, padding='SAME', normalizer_fn=tcl.batch_norm, scope='conv4')
            
            #net = tcl.max_pool2d(net,kernel_size=2)
            #net = tcl.dropout(net, keep_prob=0.8)

            #print(net.shape)
            net = tcl.flatten(net) #feature
            print(net.shape)
            feature = tcl.fully_connected(net, feature_size, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='feature')

            decoder = tcl.fully_connected(feature, 1024, activation_fn=None, normalizer_fn=tcl.batch_norm, scope='fc')
            print(decoder.shape)
            decoder = tf.reshape(decoder, (-1, 4, 4, 64))

            #decoder = tcl.conv2d_transpose(decoder, 16, 3, stride=2, # size*4
            #                        activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 64, 3, stride=1, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv1')
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 32, 3, stride=2, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv2')
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 32, 3, stride=1, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv3')
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 16, 3, stride=2, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv4')
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 8, 3, stride=1, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv5')
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, 4, 3, stride=2, # size*8
                                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv6')
            print(decoder.shape)
            x_out = tcl.conv2d_transpose(decoder, 3, 3, stride=2, # size*16
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), scope='dconv7')
            print(x_out.shape)

            #mse = tf.reduce_mean(tf.reduce_sum((inputs - x_out)**2, 1))
            mse = tf.reduce_mean(tf.abs(inputs - x_out))
            
            fc = tcl.fully_connected(feature, 2, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm, scope='fc_out')

            return feature, fc, mse, x_out
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
 
class net_in_net(object):
    def __init__(self, class_num):
        self.name = 'net_in_net'
        self.class_num = class_num

    def __call__(self, inputs, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            # Conv1 (Input Size: 128x128)
            if reuse:
                scope.reuse_variables()
            net = slim.conv2d(inputs, 192, [5, 5], padding='SAME', scope='conv1',is_training=is_training)
            net = slim.conv2d(net, 160, [1, 1], padding='SAME', scope='conv1.1',is_training=is_training)
            net = slim.conv2d(net, 96, [1, 1], padding='SAME', scope='conv1.2',is_training=is_training)
            net = slim.max_pool2d(net, [3, 3], padding='SAME', stride=2, scope='maxpool1',is_training=is_training)
            
            # Conv2 (Input Size: 64x64)
            net = slim.conv2d(net, 192, [5, 5], padding='SAME', scope='conv2',is_training=is_training)
            net = slim.conv2d(net, 192, [1, 1], padding='SAME', scope='conv2.1',is_training=is_training)
            net = slim.conv2d(net, 192, [1, 1], padding='SAME', scope='conv2.2',is_training=is_training)
            net = slim.avg_pool2d(net, [3, 3], padding='SAME', stride=2, scope='maxpool2',is_training=is_training)
            
            # Conv3 (Input Size: 32x32)
            net = slim.conv2d(net, 192, [5, 5], padding='SAME', scope='conv3',is_training=is_training)
            net = slim.conv2d(net, 192, [1, 1], padding='SAME', scope='conv3.1',is_training=is_training)
            net = slim.conv2d(net, 10, [1, 1], padding='SAME', scope='conv3.2',is_training=is_training)
            net = slim.avg_pool2d(net, [3, 3], padding='SAME', stride=2, scope='maxpool3',is_training=is_training)
            
            # Reshape
            net = tcl.flatten(net)

            # Fc1 (Input Size: 8192, OutputSize: 256)
            net = slim.fully_connected(net, 256, scope='fc1')

            # Fc2 (Input Size: 256, OutputSize: 2)
            net = slim.fully_connected(net, self.class_num, scope='fc2_Output')

            return net
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

###############################################  mlp #############################################
class G_mlp(object):
    def __init__(self):
        self.name = 'G_mlp'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64*64*3, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, tf.stack([tf.shape(z)[0], 64, 64, 3]))
            return g
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            d = tcl.fully_connected(tf.flatten(x), 64, activation_fn=tf.nn.relu,normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            logit = tcl.fully_connected(d, 1, activation_fn=None)

        return logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#-------------------------------- MNIST for test ------
class G_mlp_mnist(object):
    def __init__(self):
        self.name = "G_mlp_mnist"
        self.X_dim = 784

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
            
        return d, q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist_BEGAN():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            #mse = tf.reduce_mean(tf.reduce_sum((x - d)**2, 1))
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
            
        return d, q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Q_mlp_mnist():
    def __init__(self):
        self.name = "Q_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
        return q

    @property
    def vars(self):        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


###############################################  conv #############################################
class G_conv(object):
    def __init__(self, size, is_tanh=False):
        self.name = 'G_conv'
        self.size = size//16 #64//16
        self.channel = 3 #self.data.channel
        self.is_tanh = is_tanh
    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, self.size * self.size * 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm) #original 1024
            g = tf.reshape(g, (-1, self.size, self.size, 64))  # size
            g = tcl.conv2d_transpose(g, 32, 3, stride=2, # size*2
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 16, 3, stride=2, # size*4
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 8, 3, stride=2, # size*8
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            if self.is_tanh:
                g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                        activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            else:
                g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
    def __init__(self, size):
        self.name = 'D_conv'
        self.size = size  #64
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.flatten(shared)
    
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            # q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            # q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d#, q
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv_condition(object):
    def __init__(self, size):
        self.name = 'D_conv_cond'
        self.size = size  #64
    def __call__(self, x, y, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tf.concat([tcl.flatten(shared),y],1)
    
            d = tcl.fully_connected(shared, 256, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(d, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            #q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d#, q
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_conv_BEGAN(object):
    def __init__(self, size):
        self.name = 'G_conv'
        self.size = size//16 #64//16
        self.channel = 1 #self.data.channel
    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            g = tcl.fully_connected(z, self.size * self.size * 64, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm) #original 1024
            g = tf.reshape(g, (-1, self.size, self.size, 64))  # size
            g = tcl.conv2d_transpose(g, 32, 3, stride=2, # size*2
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 16, 3, stride=2, # size*4
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 8, 3, stride=2, # size*8
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class VAutoencoder(object):
    def __init__(self, size, feature_size):
        self.name = 'autoencoder'
        self.size = size
        self.feature_size = feature_size
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
           
            encoder = tcl.conv2d(x, num_outputs=self.size, kernel_size=3, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=tf.nn.elu)
            encoder = tcl.conv2d(encoder, num_outputs=self.size * 4, kernel_size=3, # 16x16x128
                        stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            encoder = tcl.conv2d(encoder, num_outputs=self.size * 8, kernel_size=3, # 8x8x256
                        stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            print(encoder.shape)
            #encoder = tcl.conv2d(encoder, num_outputs=self.size * 32, kernel_size=3, # 4x4x512
            #            stride=2, activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm)
            flatten = tcl.flatten(encoder)
            print(flatten.shape)
    
            feature = tcl.fully_connected(flatten, self.feature_size, weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(feature.shape)
            
            fc = tcl.fully_connected(feature, self.size//8 * self.size//8 * self.size * 8, weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(fc.shape)
            decoder = tf.reshape(fc, (-1, self.size//8, self.size//8, self.size * 8))  # size
            print(decoder.shape)
            #decoder = tcl.conv2d_transpose(decoder, 32, 3, stride=2, # size*2
            #                        activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            decoder = tcl.conv2d_transpose(decoder, self.size * 4, 3, stride=2, # size*4
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(decoder.shape)
            decoder = tcl.conv2d_transpose(decoder, self.size, 3, stride=2, # size*8
                                    activation_fn=tf.nn.elu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(decoder.shape)
            x_out = tcl.conv2d_transpose(decoder, 3, 3, stride=2, # size*16
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            print(x_out.shape)
            #x_out = tf.reshape(decoder, (-1, self.size, self.size, 3)) 

            mse = tf.reduce_mean(tf.reduce_sum((x - x_out)**2, 1))

            return mse, feature#, 0
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv(object):
    def __init__(self, size, class_num):
        self.name = 'C_conv'
        self.class_num = class_num
        self.size = size

    def __call__(self, x, is_training, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #size = 64
            shared = tcl.conv2d(x, num_outputs=self.size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
            #            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            
            q = tcl.fully_connected(tcl.flatten(shared), 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, self.class_num, activation_fn=None) # 10 classes
        
            return q
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class V_conv(object):
    def __init__(self):
        self.name = 'V_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=3, # 4x4x512
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            
            v = tcl.fully_connected(tcl.flatten(shared), 128)
            return v
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# -------------------------------- MNIST for test
class G_conv_mnist(object):
    def __init__(self, is_tanh):
        self.name = 'G_conv_mnist'
        self.is_tanh = is_tanh

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            #g = tcl.fully_connected(z, 1024, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                        weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 7*7*64, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tf.reshape(g, (-1, 7, 7, 64))  # 7x7
            g = tcl.conv2d_transpose(g, 32, 4, stride=2, # 14x14x64
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            if self.is_tanh:
                g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
                                    activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            else:
                g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
                                    activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class D_conv_mnist(object):
    def __init__(self):
        self.name = 'D_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 32
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            
            d = tcl.fully_connected(shared, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d, q
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv_mnist_BEGAN(object):
    def __init__(self):
        self.name = 'D_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 32
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            shared = tcl.fully_connected(shared, 28*28, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            x_out = tf.reshape(shared, (-1, 28, 28, 1)) 

            mse = tf.reduce_mean(tf.reduce_sum((x - x_out)**2, 1))
            q = tcl.fully_connected(shared, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return mse, q
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv_mnist(object):
    def __init__(self):
        self.name = 'C_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=5, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu)
            
            #c = tcl.fully_connected(shared, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            c = tcl.fully_connected(shared, 10, activation_fn=None) # 10 classes
            return c
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
