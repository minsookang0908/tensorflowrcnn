import cPickle
import jedi
from tflearn.layers.core import input_data, dropout, fully_connected
import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers, utils
import tflearn
from tflearn import get_training_mode
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.data_utils import shuffle, to_categorical
from tflearn.datasets import cifar100
from tflearn.data_preprocessing import ImagePreprocessing
import urllib


class RCNN:
    def __init__(self, time, K, p, num_class, is_training=True):
        self.time = time
        self.K = K
        self.p = p
        self.numclass = num_class
        self.is_training = is_training

    def mode(self, is_training):
        self.is_training = is_training

    def RCL(self, x, y, z):

        with slim.arg_scope([slim.conv2d], padding="SAME",
                            weights_initializer=initializers.xavier_initializer(),
                            activation_fn=None):
            conv1 = slim.conv2d(x, self.K, 3, scope='feedconv')

            conv1 = tf.nn.relu(conv1)
            conv1 = slim.batch_norm(conv1, is_training=self.is_training, scope='batch1')
           #  listconv = conv1.get_shape().as_list()
            #  print ('listconv' , listconv)
            feedconv2 = slim.conv2d(y, self.K, 3, scope='feedconv', reuse=True)
            reconv2 = slim.conv2d(conv1, self.K, 3, scope='reconv', biases_initializer=None)
            conv2 = tf.add(feedconv2, reconv2)
            conv2 = tf.nn.relu(conv2)
            conv2 = slim.batch_norm(conv2, is_training=self.is_training, scope='batch2')
            feedconv3 = slim.conv2d(z, self.K, 3, scope='feedconv', reuse=True)
            reconv3 = slim.conv2d(conv2, self.K, 3, scope='reconv', biases_initializer=None, reuse=True)
            conv3 = tf.add(feedconv3, reconv3)
            conv3 = tf.nn.relu(conv3)
            conv3 = slim.batch_norm(conv3, is_training=self.is_training, scope='batch3')
            print conv1.name, conv2.name, conv3.name
            return conv1, conv2, conv3

    def build_model(self, x):

        with tf.variable_scope('layer1') as scpoe:
            output = slim.conv2d(x, 96, 5, scope='conv1')
            output = tf.nn.relu(output)
            output = slim.max_pool2d(output, [2, 2], scope='pool1')
            output = slim.batch_norm(output, scope='batch_norm1', is_training=self.is_training)
        for i in range(2, 6):
            with tf.variable_scope('recurrent_conv{}'.format(i)) as scope:
                if i == 2:
                    RCL1 = self.RCL
                    output1, output2, output3 = self.RCL(output, output, output)
                # tf.nn.dropout(output1, keep_prob = 0.8)
                # tf.nn.dropout(output2, keep_prob = 0.8)
                # tf.nn.dropout(output3, keep_prob = 0.8)
                else:
                    output1, output2, output3 = self.RCL(output1, output2, output3)
                if i == 3:
                    output1 = slim.max_pool2d(output1, [2, 2], scope='pool1')
                    output2 = slim.max_pool2d(output2, [2, 2], scope='pool2')
                    output3 = slim.max_pool2d(output3, [2, 2], scope='pool3')

                if i < 5:
                    output1 = tf.layers.dropout(output1, rate=0.2, training=self.is_training, name='drop1')
                    output2 = tf.layers.dropout(output2, rate=0.2, training=self.is_training, name='drop2')
                    output3 = tf.layers.dropout(output3, rate=0.2, training=self.is_training, name='drop3')

                    print output1.name, output2.name, output3.name
        with tf.variable_scope('global_max_pooling'):
            output = slim.max_pool2d(output3, [8, 8], scope='globalpool')
            # print output.name
            output = slim.flatten(output)

        with tf.variable_scope('softmax'):
            output = slim.fully_connected(inputs=output, num_outputs=100, scope='fc')
            output = tf.nn.softmax(output)
        # print output
        return output


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


cifar100_test = {}
cifar100_train = {}
labelMap = {}
labelNames = {}
cifar100_test = unpickle('cifar-100-python/test')
cifar100_train = unpickle('cifar-100-python/train')
labelMap = unpickle('cifar-100-python/meta')
Xtr = cifar100_train[b'data']
Yr = cifar100_train[b'fine_labels']
Xte = cifar100_test[b'data']
Ye = cifar100_test[b'fine_labels']
classNames = labelMap[b'fine_label_names']
num_train = Xtr.shape[0]
num_test = Xte.shape[0]

num_class = len(classNames)

Ytr = np.zeros([num_train, num_class])
Yte = np.zeros([num_test, num_class])

# print Yr.shape
# print Yr[0]
# print Yr
# print (min(Yr))
Ytr[range(num_train), Yr] = 1
# print Ytr[0]
Yte[0:num_test, Ye[0:num_test]] = 1
num_train = Xtr.shape[0]
num_test = Xte.shape[0]

# print num_train
Xtrain = Xtr
Xtest = Xte
Ytrain = Ytr
Ytest = Yte

Xtrain = np.reshape(Xtrain, (50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(float)
Xtrain = Xtrain / 255
mean_trainB = np.mean(Xtrain[:, :, :, 0], axis=0)
mean_trainB = np.expand_dims(mean_trainB, axis=0)
mean_trainB = np.expand_dims(mean_trainB, axis=3)
mean_trainG = np.mean(Xtrain[:, :, :, 1], axis=0)
mean_trainG = np.expand_dims(mean_trainG, axis=0)
mean_trainG = np.expand_dims(mean_trainG, axis=3)
mean_trainR = np.mean(Xtrain[:, :, :, 2], axis=0)
mean_trainR = np.expand_dims(mean_trainR, axis=0)
mean_trainR = np.expand_dims(mean_trainR, axis=3)
mean = np.concatenate((mean_trainB, mean_trainG, mean_trainR), axis=3)

Xtrain = Xtrain - mean

Xtest = np.reshape(Xtest, (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(float)
Xtest = Xtest / 255
Xtest = Xtest - mean
print Xtest.shape

XtestB = np.split(Xtest, 100);
YtestB = np.split(Ytest, 100);


with tf.device('/device:GPU:0'):
# X = tf.placeholder(tf.float32, [100, 32, 32, 3])  #1->100

    Y_ = tf.placeholder(tf.float32, [100, 100])  # 1-> 100

    rcnn = RCNN(4, 96, 0.2, 100, True)
    X = input_data(shape=[None, 32, 32, 3])
    Y = rcnn.build_model(X)
    model = tflearn.DNN(Y, tensorboard_verbose=0)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), axis=1))  # normalized for batches of 100 images,
# *10 because  "mean" included an unwanted division by 10
    learning_schedule = tf.placeholder(tf.float32)
# accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct = accuracy * 100

    train_step = tf.train.GradientDescentOptimizer(learning_schedule).minimize(cross_entropy)

    with tf.Session() as sess:
      init = tf.global_variables_initializer()
    # sess = tf.Session()
        sess.run(init)

        lr = 0.01
        lr_threshold = 0.01 * 1 / 1000
        epoch = 100
        best_accuracy = 0
        count = 0
        threshold = 0.0001
        for e in range(epoch):
            permutation = np.random.permutation(50000)
        # print permutation.shape
            xtr = Xtrain.copy()
            ytr = Ytrain.copy()
            Xtrain_ = xtr[permutation]
            Ytr_ = ytr[permutation]
            Xbatches = np.split(Xtrain_, 500)  # 50000-> 500
            Ybatches = np.split(Ytr_, 500)  # 50000->500
        #     print Xbatches[0]
        # Xbatches = np.split(Xtrain, 100); second number is # of batches
        # Ybatches = np.split(np.asarray(Ytrain), 100);
            Correct = 0

            for i in range(500):  # 50000->500
                rcnn.mode(True)
                t, Loss = sess.run([train_step, cross_entropy],
                               feed_dict={X: Xbatches[i], Y_: Ybatches[i], learning_schedule: lr})
                print(Loss, i)

                if i >= 399:
                    rcnn.mode(False)
                    Correct1, t, Loss = sess.run([correct, train_step, cross_entropy],
                                             feed_dict={X: Xbatches[i], Y_: Ybatches[i], learning_schedule: lr})
                    print(Loss, i)

                    Correct += Correct1
            present_accurcy = 100.0 * float(Correct) / 10000
            if best_accuracy * (1 + threshold) < present_accurcy:
                best_accuracy = present_accurcy
                count = 0
            else:
                count += 1
    
            if count == 4:
                if lr > lr_threshold:
                    lr = lr / 10
                    count = 0
                else:
                    count = 0
                    lr = lr_threshold

        for i in range(100):
            print('accuracy:', sess.run(accuracy, feed_dict={X: XtestB[i], Y_: YtestB[i]}))
