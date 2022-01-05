#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:52:42 2019

@author: gdisciac
"""
## DiffNet Script with only one CNN with 9 inputs to get Gamma
## 9-POINT STENCIL

import tensorflow as tf
import sys

sys.path.append("..")
import Load_DiffNet_Inv as loadDiff
import numpy as np
import utilTemp_nonlinInv_mult as util
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import scipy.io as sio  # %ok- 1
from random import randint
# from numba import cuda
import multiprocessing

# import time

print('pausing')
# time.sleep(3600*2)
print('resuming')
# import matplotlib.pyplot as plt

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

bSize = int(16)
chan = int(1)


fDepth = int(5);
N = int(60)
NN = N * N

zero = tf.constant(0.0, shape=[bSize, 1])
zeroNHorz = tf.constant(0.0, shape=[bSize, N, 1, 1])
zeroNVert = tf.constant(0.0, shape=[bSize, 1, N, 1])
zeroNN = tf.constant(0.0, shape=[bSize, N, N,1])


def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# dropOut=0.01
def kappaEstimator(x_in):
    x_in = tf.reshape(x_in, [bSize, N, N, 1])
    kappaEst = tf.contrib.layers.conv2d(x_in, 32, 3)
    kappaEst = tf.contrib.layers.conv2d(kappaEst, 32, 3)
    kappaEst = tf.contrib.layers.conv2d(kappaEst, 32, 3)

    kappaEst = tf.contrib.layers.conv2d(kappaEst, 5, 3, activation_fn=None)

    return kappaEst


def weight_variable(shape, layernum ,layerN ):
#   initial = tf.truncated_normal(shape, stddev=0.005)
    x =  np.sqrt(2/(shape[0]*shape[1]*shape[2]*(shape[3]+shape[4]+ layerN)));
    #initializer = tf.random_uniform(shape, minval=-x, maxval=x);
    initializer = tf.truncated_normal(shape, stddev= x);
    return tf.get_variable(layernum, initializer=initializer)


def bias_variable(shape):
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x, W):
    """conv3d returns a 3d convolution layer with full stride."""
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def d_conv2d(x, W, output_shape):
    """conv3d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')

def d_conv3d(x, W, output_shape):
    """conv3d returns a 2d convolution layer with full stride."""
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 1, 1, 1, 1], padding='SAME')

def u_net_step_down(x, w1, w2, w3):
    l1 = tf.nn.relu(conv2d(x, w1));
    l2 = tf.nn.relu(conv2d(l1, w2));
    l3 = conv2d(l2, w3);
    out = tf.nn.relu(tf.add(l3, l1));
    return out;


def u_net_step_up(x, w1, w2, w3, output_shape):
    l1 = tf.nn.relu(d_conv2d(x, w1, output_shape));
    l2 = tf.nn.relu(conv2d(l1, w2));
    l3 = conv2d(l2, w3);
    out = tf.nn.relu(tf.add(l3, l1));
    return out;


def diffLayer(x_in, bSize, N, layNum):
    #    NN=N*N

    x_full = tf.reshape(x_in, [bSize, N, N, chan])
    x_update = x_full[:, :, :, 0]
    x_update = tf.reshape(x_update, [bSize, N, N, 1])
    Kappa = [];
    dt = tf.constant(0.1, shape=[1], dtype=tf.float32)
    dt = tf.Variable(dt, name='dt_' + str(layNum));
    # First network initialises variables
    # kappa=kappaEstimator(x_update)
    with tf.name_scope('NetworkInit'):
        #        kappaEst=tf.contrib.layers.conv2d(x_in,32,3)
        W_convL1 = weight_variable([3, 3, chan, 32])
        b_convL1 = bias_variable([32])
        kappaEst = tf.nn.relu(conv2d(x_full, W_convL1) + b_convL1)

        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL2 = weight_variable([3, 3, 32, 32])
        b_convL2 = bias_variable([32])
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL2) + b_convL2)

        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL3 = weight_variable([3, 3, 32, 32])
        b_convL3 = bias_variable([32])
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL3) + b_convL3)

        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)

        W_convL4 = weight_variable([3, 3, 32, 9])
        b_convL4 = bias_variable([9])
        kappa = (conv2d(kappaEst, W_convL4) + b_convL4)
        Kappa.append(kappa[:, :, :, 0:8]);

    kapDiag = tf.reshape(kappa[:, :, :, 0], [bSize, N, N, 1])
    kapUp = tf.reshape(kappa[:, :, :, 1], [bSize, N, N, 1])
    kapDown = tf.reshape(kappa[:, :, :, 2], [bSize, N, N, 1])
    kapLeft = tf.reshape(kappa[:, :, :, 3], [bSize, N, N, 1])
    kapRight = tf.reshape(kappa[:, :, :, 4], [bSize, N, N, 1])
    kapUpRight = tf.reshape(kappa[:, :, :, 5], [bSize, N, N, 1])
    kapUpLeft = tf.reshape(kappa[:, :, :, 6], [bSize, N, N, 1])
    kapDownRight = tf.reshape(kappa[:, :, :, 7], [bSize, N, N, 1])
    kapDownLeft = tf.reshape(kappa[:, :, :, 8], [bSize, N, N, 1])

    # b_dt0 = bias_variable([1])
    # b_dt = bias_variable([1])
    # dt = tf.reshape(tf.reduce_mean(tf.reshape(kappa[:,:,:,9],[bSize,N,N,1] ), [ 1,2,3]), [bSize,1,1,1]))#tf.tile[1,N,N,1]);

    xUp = tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)
    xDown = tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)
    xLeft = tf.concat([zeroNHorz, x_update[:, :, 0:N - 1]], axis=2)
    xRight = tf.concat([x_update[:, :, 1:N], zeroNHorz], axis=2)
    xUpRight = tf.concat([tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)[:, :, 1:N], zeroNHorz], axis=2)
    xDownLeft = tf.concat([zeroNHorz, tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)[:, :, 0:N - 1]], axis=2)
    xUpLeft = tf.concat([zeroNHorz, tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)[:, :, 0:N - 1]], axis=2)
    xDownRight = tf.concat([tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)[:, :, 1:N], zeroNHorz], axis=2)

    xDiag = tf.multiply(kapDiag, x_update)
    xUp = tf.multiply(kapUp, xUp)
    xDown = tf.multiply(kapDown, xDown)
    xLeft = tf.multiply(kapLeft, xLeft)
    xRight = tf.multiply(kapRight, xRight)
    xUpRight = tf.multiply(kapUpRight, xUpRight)
    xDownLeft = tf.multiply(kapDownLeft, xDownLeft)
    xUpLeft = tf.multiply(kapUpLeft, xUpLeft);
    xDownRight = tf.multiply(kapDownRight, xDownRight);

    # x_update = x_update + tf.scalar_mul(dt, (xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight) ) #Test +xDiag
    x_update = x_update + tf.multiply(dt, (
                xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight))  # Test +xDiag
    x_out = tf.reshape(x_update, [bSize, N, N, 1])

    for xi in range(chan - 1):
        x_update = x_full[:, :, :, xi + 1]
        x_update = tf.reshape(x_update, [bSize, N, N, 1])
        # update for all channels with same kappa found before
        xUp = tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)
        xDown = tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)
        xLeft = tf.concat([zeroNHorz, x_update[:, :, 0:N - 1]], axis=2)
        xRight = tf.concat([x_update[:, :, 1:N], zeroNHorz], axis=2)
        xUpRight = tf.concat([tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)[:, :, 1:N], zeroNHorz], axis=2)
        xDownLeft = tf.concat([zeroNHorz, tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)[:, :, 0:N - 1]], axis=2)
        xUpLeft = tf.concat([zeroNHorz, tf.concat([zeroNVert, x_update[:, 0:N - 1, :]], axis=1)[:, :, 0:N - 1]], axis=2)
        xDownRight = tf.concat([tf.concat([x_update[:, 1:N, :], zeroNVert], axis=1)[:, :, 1:N], zeroNHorz], axis=2)

        xDiag = tf.multiply(kapDiag, x_update)
        xUp = tf.multiply(kapUp, xUp)
        xDown = tf.multiply(kapDown, xDown)
        xLeft = tf.multiply(kapLeft, xLeft)
        xRight = tf.multiply(kapRight, xRight)
        xUpRight = tf.multiply(kapUpRight, xUpRight)
        xDownLeft = tf.multiply(kapDownLeft, xDownLeft)
        xUpLeft = tf.multiply(kapUpLeft, xUpLeft);
        xDownRight = tf.multiply(kapDownRight, xDownRight);

        # x_update = x_update + tf.scalar_mul(dt,(xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight)) #Test +xDiag
        x_update = x_update + tf.multiply(dt, (
                    xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight))  # Test +xDiag

        x_out = tf.concat([x_out, x_update], axis=3)

    #    diagFac = getKappa(4.0, [N*N],'diagFac_' + str(layNum))
    #    xDiag=tf.reshape(xDiag,[bSize,N*N])
    #    xDiag  = tf.multiply(diagFac,xDiag)
    #    xDiag=tf.reshape(xDiag,[bSize,N,N,1])

    return x_out, kappa, Kappa

def diffLayer3D(x_in, bSize, N, fDepth, layNum):
    #    NN=N*N

    x_full = tf.reshape(x_in, [bSize, N, N,fDepth, chan])
    x_update = x_full[:, :, :, : , 0]
    x_update = tf.reshape(x_update, [bSize, N, N,fDepth, 1])
    Kappa = [];
    dt = tf.constant(0.1, shape=[1], dtype=tf.float32)
    dt = tf.Variable(dt, name='dt_' + str(layNum));
    # First network initialises variables
    # kappa=kappaEstimator(x_update)
    with tf.name_scope('NetworkInit'):
        #        kappaEst=tf.contrib.layers.conv2d(x_in,32,3)
        W_convL1 = weight_variable([3, 3, 3, chan, 32], 'Wconv_1_'+str(layNum),   1 +layNum - 10)
        b_convL1 = bias_variable([32])
        W_dconvL1 = weight_variable([3, 3, 3, 32, 32], 'Wconv_1_' + str(layNum) + 'dconv',1 +layNum - 10)
        kappaEst = tf.nn.relu(conv3d(x_full, W_convL1) + b_convL1)
        kappaEst = tf.nn.max_pool3d(kappaEst, [1,2,2,2,1], [1,1,1,1,1],'SAME')
        kappaEst = d_conv3d(kappaEst, W_dconvL1,kappaEst.get_shape())

        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL2 = weight_variable([3, 3, 3, 32, 32], 'Wconv_2_'+str(layNum), 1+layNum - 10)
        b_convL2 = bias_variable([32])
        W_dconvL2 = weight_variable([3, 3, 3, 32, 32], 'Wconv_2_' + str(layNum) + 'dconv',1 +layNum - 10)
        kappaEst = tf.nn.relu(conv3d(kappaEst, W_convL2) + b_convL2)
        kappaEst = tf.nn.max_pool3d(kappaEst, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], 'SAME')
        kappaEst = d_conv3d(kappaEst, W_dconvL2, kappaEst.get_shape())

        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL3 = weight_variable([3, 3, 3, 32, 32], 'Wconv_3_'+str(layNum), 1+layNum - 10)
        W_dconvL3 = weight_variable([3, 3, 3, 32, 32], 'Wconv_3_' + str(layNum) + 'dconv', 1 +layNum - 10)
        b_convL3 = bias_variable([32])
        kappaEst = tf.nn.relu(conv3d(kappaEst, W_convL3) + b_convL3)
        kappaEst = tf.nn.max_pool3d(kappaEst, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], 'SAME')
        kappaEst = d_conv3d(kappaEst, W_dconvL3, kappaEst.get_shape())

        W_convL4 = weight_variable([3, 3, 3, 32, 27], 'Wconv_4_'+str(layNum), 1+ layNum - 10)
        b_convL4 = bias_variable([27])
        kappa = (conv3d(kappaEst, W_convL4) + b_convL4)
        Kappa.append(kappa[:, :, :,:, 0:27]);

    x_update = x_update + tf.multiply(dt, assembleXupdate3D(x_update,kappa)); #Test +xDiag
    x_out = tf.reshape(x_update, [bSize, N, N, fDepth, 1])

    for xi in range(chan - 1):
        x_update = x_full[:, :, :, :, xi + 1]
        x_update = tf.reshape(x_update, [bSize, N, N,fDepth, 1])
        # update for all channels with same kappa found before
        x_update = x_update + tf.multiply(dt, assembleXupdate3D(x_update, kappa));  # Test +xDiag
        x_out = tf.reshape(x_update, [bSize, N, N, fDepth, 1])

        x_out = tf.concat([x_out, x_update], axis=3)

    return x_out, kappa, Kappa
    #    diagFac = getKappa(4.0, [N*N],'diagFac_' + str(layNum))
    #    xDiag=tf.reshape(xDiag,[bSize,N*N])
    #    xDiag  = tf.multiply(diagFac,xDiag)
    #    xDiag=tf.reshape(xDiag,[bSize,N,N,1])

def assembleXupdate3D(x_update,kappa):
    count_k = 0;
    x_out = tf.zeros_like(x_update);
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1 ,2):
                x_out = x_out + tf.multiply(tf.reshape(kappa[:,:,:,:,count_k], [bSize, N,N,fDepth,1]), shiftX(x_update, i,j,k));
                count_k = count_k + 1;
    #print(count_k)
    return x_out

def shiftX(xupdate,i,j,k):

    ## shifts xupdate in order to be multiplied fo kappa in a second time
    x_shifted = tf.pad( tf.reshape(xupdate, [bSize, N, N,fDepth, 1]), [[0,0],[1,1],[1,1],[1,1],[0,0]]);
    #x_shifted = tf.manip.roll(x_shifted, i, axis = 1);
    #x_shifted = tf.manip.roll(x_shifted, j, axis = 2);
    #x_shifted = tf.manip.roll(x_shifted, k, axis = 3);
    x_shifted = tf.manip.roll(x_shifted, [i,j,k], axis=[1,2,3]);
    x_out = tf.reshape(x_shifted[:,1:N+1,1:N+1,1:fDepth+1, :], [bSize, N,N,fDepth,1]);
    return x_out

def diffLayer_channel(x_in, bSize, N, chan):
    x_chan = tf.expand_dims(diffLayer(x_in[:, :, :, 0], bSize, N), 2)

    for bbb in range(chan - 1):
        x_chan = tf.concat([x_chan, tf.expand_dims(diffLayer(x_in[:, :, :, bbb + 1], bSize, N), 2)], axis=2)

    x_chan = tf.reshape(x_chan, [bSize, N, N, chan])
    return x_chan


def getKappa(inVal, shape, varName):
    kappa = tf.constant(inVal, shape=shape)
    return tf.Variable(kappa, name=varName)


def main(filePath, fileOutName, dataSetTrain, dataSetTest, tRand, MatOutName):
    iterMain = int(1)
    maxIter = int(30000)
    print('--------------------> DiffNet Init <--------------------')
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    dataDbar = [None]*len(dataSetTest);
    for i_datasets in range(0, len(dataSetTest) ):
        dataDbar[i_datasets] = loadDiff.read_data_sets(dataSetTrain, dataSetTest[i_datasets]);
    imSize = dataDbar[0].train.true.shape

    # Create the model
    imag = tf.placeholder(tf.float32, [bSize, imSize[1], imSize[2], chan])
    true = tf.placeholder(tf.float32, [bSize, imSize[1], imSize[2],1])
    x_update = tf.zeros_like(imag);
    with tf.name_scope('DiffNet'):
        x_update3d = (tf.reshape(tf.concat([tf.tile( tf.expand_dims(tf.reshape(imag, [bSize,N,N,chan]),3), [1,1,1,1,1]), tf.zeros([bSize, N,N,fDepth-1, chan])], axis = 3), [bSize, N, N,fDepth, chan]))
        iii = 0;
        for iii in range(iterMain):
            x_update3d, kappaEst1, Kappa1 = diffLayer3D(x_update3d, bSize, N, fDepth, 0 + 10 * iii)
            x_update3d, kappaEst2, Kappa2 = diffLayer3D(x_update3d, bSize, N, fDepth, 1 + 10 * iii)
            x_update3d, kappaEst3, Kappa3 = diffLayer3D(x_update3d, bSize, N, fDepth, 2 + 10 * iii)
            x_update3d, kappaEst4, Kappa4 = diffLayer3D(x_update3d, bSize, N, fDepth, 3 + 10 * iii)
            x_update3d, kappaEst5, Kappa5 = diffLayer3D(x_update3d, bSize, N, fDepth, 4 + 10 * iii)
            x_update3d, kappaEst5, Kappa5 = diffLayer3D(x_update3d, bSize, N, fDepth, 5 + 10 * iii)
            x_update = x_update + tf.reshape(x_update3d[:, :, :, fDepth - 3, :], [bSize, N, N, chan])
 #           x_update3d, _, _ = diffLayer3D(x_update3d, bSize, N, fDepth, 5 + 10 * iii)
 #           x_update = x_update + tf.reshape(x_update3d[:, :, :, fDepth - 1, :], [bSize, N, N, chan])
#        x_update = tf.reshape(x_update3d[:,:,:,fDepth - 1, :], [bSize, N, N,chan])
        x_sum = x_update[:, :, :, 0]
        for ccc in range(chan - 1):
            x_sum = x_sum + x_update[:, :, :, ccc + 1]
        x_sum = x_sum / chan
        y_diff = tf.nn.tanh(tf.nn.relu(x_sum));
        y_diff = tf.reshape(y_diff, [bSize,N,N, 1]);
    saver = tf.train.Saver()

    with tf.name_scope('optimizer'):
        loss = tf.norm(tf.subtract(tf.nn.relu(true), y_diff)) / float(bSize)
        rel_loss = tf.norm(tf.subtract(tf.nn.relu(true), y_diff)) / tf.norm(tf.nn.relu(true))
        all_weights = tf.trainable_variables()
        regularizer = 0.000# * tf.add_n( [ tf.nn.l1_loss(v_weights) for v_weights in all_weights] );

        # rel_loss = tf.norm( tf.divide(tf.subtract(tf.nn.relu(true),y_diff),(0.01 + tf.nn.relu(true))))
        #     added_loss = -tf.scalar_mul(100.0,tf.minimum( tf.subtract(tf.norm(y_diff),10.0),0.0))
        #    global_step = tf.Variable(0, trainable=False)


    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('rel_loss', rel_loss)
        tf.summary.scalar('psnr', psnr(y_diff, true))
        tf.summary.image('diffused', tf.reshape(imag[1], [1, imSize[1], imSize[2], 1]))
        tf.summary.image('result', tf.reshape(y_diff[1], [1, imSize[1], imSize[2], 1]))
        #tf.summary.image('kappaEst1', tf.reshape(kappaEst1[1, :, :, 0], [1, imSize[1], imSize[2], 1]))
        #tf.summary.image('kappaEst2', tf.reshape(kappaEst2[1, :, :, 0], [1, imSize[1], imSize[2], 1]))
        tf.summary.image('x_channel1', tf.nn.relu(tf.reshape(x_update[1, :, :, 0], [1, imSize[1], imSize[2], 1])))
        # tf.summary.image('x_channel2', tf.nn.relu(tf.reshape(x_update[1,:,:,1],[1, imSize[1], imSize[2], 1]) ))
        # tf.summary.image('x_channel3', tf.nn.relu(tf.reshape(x_update[1,:,:,2],[1, imSize[1], imSize[2], 1]) ))
        # tf.summary.image('x_channel4', tf.nn.relu(tf.reshape(x_update[1,:,:,3],[1, imSize[1], imSize[2], 1]) ))
        #    tf.summary.image('residual', (true - primal_result_pos)[..., zBatch // 2: zBatch // 2 + 1])
        tf.summary.image('true', tf.reshape(true[1], [1, imSize[1], imSize[2], 1]))
        #        tf.summary.image('imag', tf.reshape(imag[1,:,:,0:4],[1, imSize[1], imSize[2], 4]) )
        tf.summary.image('diff', tf.reshape(true[1] - y_diff[1], [1, imSize[1], imSize[2], 1]))
        expName = [None]*len(fileOutName);
        for i_expName in range(0, len(fileOutName)):
            expName[i_expName] = 'DiffNet_' + fileOutName[i_expName];

        test_summary_writer, train_summary_writer = util.summary_writers(name, expName, cleanup=False)


        merged_summary = tf.summary.merge_all()
    with tf.name_scope('training'):
        learningRate = tf.constant(1e-3)
        train_step = tf.train.AdamOptimizer(learningRate).minimize(rel_loss+regularizer)


    sess.run(tf.global_variables_initializer())

    feed_test = {imag: dataDbar[0].test.images[0:bSize],
                 true: dataDbar[0].test.true[0:bSize]}

    lVal = 5e-4
    startIt = 0
    for i in range(maxIter):

        batch = dataDbar[0].train.next_batch(bSize)
        #      test1 = batch[0]
        #      test2 = batch[1]
        feed_train = {imag: batch[0], true: batch[1], learningRate: lVal}

        _, merged_summary_result_train, loss_result_train = sess.run([train_step, merged_summary, rel_loss],
                                                                     feed_dict=feed_train)


        #if i == 100:
         #   lVal = lVal/2
        if i == 800:
            lVal = lVal
        if i == 900:
            lVal = lVal
        if i == 1500:
            lVal = lVal/2
        if i == 1200:
            lVal = lVal
    #    if i > 500 and i < 540:
     #       lVal = 2e-4;

        if i % 20 == 0:
            # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            tBeg = randint(0, tRand)
            tEnd = tBeg + bSize
            it = i + startIt

            for i_test in range(0,len(dataDbar)):

                feed_test = {imag: dataDbar[i_test].test.images[tBeg:tEnd],
                             true: dataDbar[i_test].test.true[tBeg:tEnd]}

                test_accuracy, test_result = sess.run([loss, y_diff],
                                                      feed_dict=feed_test)

                loss_result, rel_loss_res, merged_summary_result = sess.run([loss, rel_loss, merged_summary],
                                                                            feed_dict=feed_test)

                train_summary_writer.add_summary(merged_summary_result_train, it)
                test_summary_writer[i_test].add_summary(merged_summary_result, it)

                print('iter={}, loss={}, rel.loss.test={}, rel.loss.train={}'.format(i, loss_result, rel_loss_res,
                                                                                     loss_result_train))
    Kappa1, Kappa2, Kappa3, Kappa4, Kappa5, results = sess.run([Kappa1, Kappa2, Kappa3, Kappa4, Kappa5, x_sum],
                                                               feed_dict=feed_test);
    Kappa_list = []
    Kappa_list.append(Kappa1);
    Kappa_list.append(Kappa2);
    Kappa_list.append(Kappa3);
    Kappa_list.append(Kappa4);
    Kappa_list.append(Kappa5);
    dataDbar[0].test.images[tBeg:tEnd]
    dataDbar[0].test.true[tBeg:tEnd]
    save_path = saver.save(sess, filePath)
    print("Model saved in file: %s" % save_path)
    # get gamma and results
    dict_sio = {
        'Kappa': Kappa_list,
        'Input': dataDbar.test.images[tBeg:tEnd],
        'Output': results,
        'True': dataDbar.test.true[tBeg:tEnd]
    }
    sio.savemat(MatOutName, dict_sio)
    print('Result Saved')

    sess.close();
    #   tf.reset_default_graph()

    print('--------------------> DONE <--------------------')

folder_tensorboard = '';
folder_data = '../processed4python/multiGamma/original_diffnet/DOCM_reflection/';
commonName_train = folder_data + 'DOCM4DiffNet_reflection_liqMus';
commonName_test = folder_data + 'DOCM4DiffNet_reflection_liqMus';
specName_train = ['1','2', '_merged'];
specName_test =  [ [ '1', '2','_merged' ],['2','_merged', '1' ],['_merged', '2','1' ]];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];
#specName_train = [ '2'];
#specName_test =  [['2']];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];

#specName_train = [ 'allRandom' ,'nohete', 'from1to4' ];
#specName_test = [['allRandom','from1to4','nohete'] ,['nohete', 'from1to4', 'allRandom'],['from1to4','nohete', 'allRandom']];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];
netPath = 'netData/test_DOCM.ckpt';
train_num = np.multiply([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],150 - bSize -1)
saveMatCommon = 'RESULTS/'+'DOCM_orig_sepTest/';#implicit'
logNameCommon = folder_tensorboard +'orig_5b5filterconvDconv_sepTest_';


# outL = main(netPath,logName,dataSetTrain,dataSetTest, 1050,savematName)
for j_proc in range(0, len(specName_train)):
    dataSetTest = [None]*len(specName_test[j_proc]);
    logName = [None]*len(specName_test[j_proc]);
    savematName = [None]*len(specName_test[j_proc]);
    dataSetTrain = commonName_train + specName_train[j_proc] + '.mat';
    for i_logname in range(0, len(specName_test[j_proc])):
        dataSetTest[i_logname] = commonName_test + specName_test[j_proc][i_logname] + '_t.mat';
        logName[i_logname] = logNameCommon + '_train_' + specName_train[j_proc] + '_test_' + specName_test[j_proc][i_logname];
        savematName[i_logname] = saveMatCommon + '_train_' + specName_train[j_proc] + '_test_' + specName_test[j_proc][i_logname] + '.mat';
    print(logName)
    # Lout,output =main(netPath,logName,dataSetTrain,dataSetTest, train_num[j_proc], savematName)
    p = multiprocessing.Process(target=main,
                                args=(netPath, logName, dataSetTrain, dataSetTest, train_num[j_proc], savematName));

    p.start()
    p.join()
    p.terminate()
