"""
Simple tester for the vgg19_trainable
"""

import vgg2
import sys
import os
import time
import tensorflow as tf
import tflearn
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import threading
import importlib
import util


#hyparameter
batchsize =100
epoch_num =100
batch_num =100
lr = 0.0001
#preprocess,每一个图片变成灰度的图象，

val_img_path = "/data/srd/data/Image/ImageNet/val"
annotations = "/data/srd/data/Image/ImageNet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
## test process


images_val = sorted(os.listdir(val_img_path))
labels = util.read_test_labels(annotations)
label_predict = []
label_prob = []
top5_correct = []


batch_x = util.load_image("./test_data/tiger.jpeg")
def next_x(batchsize):
#下个batch


#onehot的y,preprocess
batch_y =
def next_y(self,batchszie):
#下一个batch


with tf.device('/gpu:0','/gpu:1'):
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [batchsize, 224, 224, 1])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)  #true
    vgg = vgg2.Vgg19()
    vgg.build(images, train_mode)

    '''
     # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
   print(vgg.get_var_count())
    '''

    # training parameter

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.prob, labels=batch_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(batch_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    #每一个batch给优化一下，每一个epoch输出一次acc,然后20个epoch保存一下vgg
    for epoch in range(epoch_num):
        for batch in range(batch_num):
            #每一个batch我才读一次数据对吧!
            true_num=0
            sess.run(optimizer, feed_dict={images: batch_x, true_out: batch_y, train_mode: True})
            batch_x=next_x(batch_x)
            batch_y=next_y(batch_y)
            if batch%20==0: print(accuracy)
        if epoch%20==0:
            vgg.save_npy(sess, './test-save.npy')
    #保存vgg
