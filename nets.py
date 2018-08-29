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
