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

sys.path.append("../")
import Load_DiffNet_Inv as loadDiff
import numpy as np
import utilTemp_nonlinInv_mult as util
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import scipy.io as sio  # %ok- 1
from random import randint
# from numba import cuda
import multiprocessing
import time

print('pausing')
time.sleep(3600 * 0)
print('resuming')
# import matplotlib.pyplot as plt

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

name = '/scratch0/NOT_BACKED_UP/gdisciac/mcx_files/tensorboard/DOCM_2020Remake/transmission/EXP/CBH/202106_CBH/DiffNet/Interpretable9/';
name = '/home/gdisciac/202106_CBH_res/DiffNet/'+'/Interpretable9mua/'+sys.argv[1]+'/';
#name = '/scratch0/NOT_BACKED_UP/gdisciac/mcx_files/tensorboard/DOCM_2020Remake/transmission/EXP/CBH/first_session/DiffNet/Interpretable9mua/';
bSize = int(1)
chan = int(1)

N = int(60)
NN = N * N
Knature ='tensorial';#'tensorial'#sys.argv[1];#'scalar';#'tensorial';#
print('KNATURE IS SET TO ' + Knature)
zero = tf.constant(0.0, shape=[bSize, 1])
zeroNHorz = tf.constant(0.0, shape=[bSize, N, 1, 1])
zeroNVert = tf.constant(0.0, shape=[bSize, 1, N, 1])


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


def weight_variable(shape, layernum):
    #   initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.get_variable(layernum, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def bias_variable(shape):
    initial = tf.constant(0.025, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv3d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def d_conv2d(x, W, output_shape):
    """conv3d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')


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


def diffLayer_old(x_in, bSize, N, layNum):
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
        chans_in = 32;
        width = 5;
        pool_w = 3
        W_convL1 = weight_variable([width, width, chan, chans_in], 'W1' + str(layNum))
        b_convL1 = bias_variable([chans_in])
        W_dconvL1 = weight_variable([width, width, chans_in, chans_in], 'W1' + str(layNum) + 'dconv')
        kappaEst_d1 = tf.nn.relu(conv2d(x_full, W_convL1) + b_convL1)
        kappaEst_d1p = tf.nn.max_pool(kappaEst_d1, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL2 = weight_variable([width, width, chans_in, chans_in], 'W2' + str(layNum))
        b_convL2 = bias_variable([chans_in])
        W_dconvL2 = weight_variable([width, width, chans_in, chans_in], 'W2' + str(layNum) + 'dconv')
        kappaEst_d2 = tf.nn.relu(conv2d(kappaEst_d1p, W_convL2) + b_convL2)
        kappaEst_d2p = tf.nn.max_pool(kappaEst_d2, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL3 = weight_variable([width, width, chans_in, chans_in], 'W3' + str(layNum))
        b_convL3 = bias_variable([chans_in])
        W_dconvL3 = weight_variable([width, width, chans_in, chans_in], 'W3' + str(layNum) + 'dconv')
        kappaEst_d3 = tf.nn.relu(conv2d(kappaEst_d2p, W_convL3) + b_convL3)
        kappaEst_d3p = tf.nn.max_pool(kappaEst_d3, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')

        W_convL4 = weight_variable([width, width, chans_in, chans_in], 'W4' + str(layNum))
        b_convL4 = bias_variable([chans_in])
        W_dconvL4 = weight_variable([width, width, chans_in, chans_in], 'W4' + str(layNum) + 'dconv')
        kappaEst_d4 = tf.nn.relu(conv2d(kappaEst_d3p, W_convL4) + b_convL4)
        kappaEst_d4p = tf.nn.max_pool(kappaEst_d4, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')

        kappaEst = tf.nn.relu(d_conv2d(kappaEst_d4p, W_dconvL4, kappaEst_d4.get_shape()) + kappaEst_d4);
        kappaEst = tf.nn.relu(d_conv2d(kappaEst, W_dconvL3, kappaEst_d3.get_shape()) + kappaEst_d3);
        kappaEst = tf.nn.relu(d_conv2d(kappaEst, W_dconvL2, kappaEst_d2.get_shape()) + kappaEst_d2);
        kappaEst = tf.nn.relu(d_conv2d(kappaEst, W_dconvL1, kappaEst_d1.get_shape()) + kappaEst_d1);
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)
        W_out = weight_variable([width, width, chans_in, 9], 'Wout' + str(layNum))
        b_out = bias_variable([9])
        kappa = tf.tile(weight_variable([1, N, N, 9], 'Wfix' + str(layNum)), [bSize, 1, 1, 1]);
        kappa = kappa + (conv2d(kappaEst, W_out) + b_out)

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


def diffLayer(x_in, bSize, N, layNum):
    #    NN=N*N
    order_step = 1;
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
        chans_in = 32;
        width = 3;
        pool_w = 2;

        W0 = weight_variable([width, width, chan, chan], 'W0' + str(layNum))
        b0 = bias_variable([chan]);
        x_full = (conv2d(x_full, W0)) + b0;
        x_update = x_full[:, :, :, 0]
        x_update = tf.reshape(x_update, [bSize, N, N, 1])

        W_convL1 = weight_variable([width, width, chan, chans_in], 'W1' + str(layNum))
        b_convL1 = bias_variable([chans_in])
        W_dconvL1 = weight_variable([width, width, chans_in, chans_in], 'W1' + str(layNum) + 'dconv')
        kappaEst_d1 = tf.nn.elu(conv2d(x_full, W_convL1) + b_convL1)
        kappaEst_d1p = tf.nn.max_pool(kappaEst_d1, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL2 = weight_variable([width, width, chans_in, chans_in], 'W2' + str(layNum))
        b_convL2 = bias_variable([chans_in])
        W_dconvL2 = weight_variable([width, width, chans_in, chans_in], 'W2' + str(layNum) + 'dconv')
        kappaEst_d2 = tf.nn.elu(conv2d(kappaEst_d1p, W_convL2) + b_convL2)
        kappaEst_d2p = tf.nn.max_pool(kappaEst_d2, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL3 = weight_variable([width, width, chans_in, chans_in], 'W3' + str(layNum))
        b_convL3 = bias_variable([chans_in])
        W_dconvL3 = weight_variable([width, width, chans_in, chans_in], 'W3' + str(layNum) + 'dconv')
        kappaEst_d3 = tf.nn.elu(conv2d(kappaEst_d2p, W_convL3) + b_convL3)
        kappaEst_d3p = tf.nn.max_pool(kappaEst_d3, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')

        W_convL4 = weight_variable([width, width, chans_in, chans_in], 'W4' + str(layNum))
        b_convL4 = bias_variable([chans_in])
        W_dconvL4 = weight_variable([width, width, chans_in, chans_in], 'W4' + str(layNum) + 'dconv')
        kappaEst_d4 = tf.nn.elu(conv2d(kappaEst_d3p, W_convL4) + b_convL4)
        kappaEst_d4p = tf.nn.max_pool(kappaEst_d4, [1, pool_w, pool_w, 1], [1, 1, 1, 1], 'SAME')

        kappaEst = tf.nn.elu(d_conv2d(kappaEst_d4p, W_dconvL4, kappaEst_d4.get_shape()) + kappaEst_d4);
        kappaEst = tf.nn.elu(d_conv2d(kappaEst, W_dconvL3, kappaEst_d3.get_shape()) + kappaEst_d3);
        kappaEst = tf.nn.elu(d_conv2d(kappaEst, W_dconvL2, kappaEst_d2.get_shape()) + kappaEst_d2);
        kappaEst = tf.nn.elu(d_conv2d(kappaEst, W_dconvL1, kappaEst_d1.get_shape()) + kappaEst_d1);
        #        kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)

        if Knature=='tensorial':
            out_chan = 7;
        elif Knature=='scalar':
            out_chan = 2;

        W_out = weight_variable([width, width, chans_in, out_chan], 'Wout' + str(layNum))
        b_out = bias_variable([out_chan])
        kappa = tf.tile(weight_variable([1, N, N, out_chan], 'Wfix' + str(layNum)), [1, 1, 1, 1]);

        kappa = kappa + (conv2d(kappaEst, W_out) + b_out)
        Kappa.append(kappa[:, :, :, 0:out_chan]);


    x_update = x_update + tf.multiply(tf.reshape(kappa[:,:,:,-1],[bSize,N,N,1]),x_update) - assembleXupdate2D(x_update, kappa[:, :, :, 0:out_chan-3], dt,
                                            order_step, str_Knature=Knature)- assembleVelocityUpdate(x_update[out_chan-3:out_chan-1], kappa, dt)
    x_out = tf.reshape(x_update, [bSize, N, N, 1])

    for xi in range(chan - 1):
        x_update = x_full[:, :, :, xi + 1]
        x_update = tf.reshape(x_update, [bSize, N, N, 1])
        # update for all channels with same kappa found before
        x_update = x_update + tf.multiply(tf.reshape(kappa[:, :, :, -1], [bSize, N, N, 1]), x_update) -
                    assembleXupdate2D(x_update, kappa[:, :, :, 0:out_chan - 3], dt,order_step,str_Knature=Knature) -
        x_out = tf.concat([x_out, x_update], axis=3)

    return x_out, kappa, Kappa

def assembleVelocityUpdate(x_update, kappa, dt):
    x_out = tf.zeros_like(x_update);
    count_k = 0;
    if str_Knature=='tensorial':
        multK = [None]*2
        multK[0] = kappa[:,:,:,0];
        multK[1] = kappa[:,:,:,1];
        dx_ =
        dy_ =
        x_out = tf.multiply(mult[0], shiftX(x_update, 2, 0)+shiftX(x_update, -2, 0)-x_updata)+
                tf.multiply(mult[1], shiftX(x_update, 0, 2)+shiftX(x_update,  0,-2)-x_update);

    x_out_mult = tf.multiply(dt, x_out);
    return x_out_mult

def assembleXupdate2D(x_update, kappa, dt, order, str_Knature='tensorial'):
    x_out = tf.zeros_like(x_update);
    count_k = 0;
    if str_Knature=='tensorial':
        multK = [None]*9
        multK[4] = -( shiftX(kappa[:,:,:,0], 0,0)+shiftX(kappa[:,:,:,3], 0,0)
                         +shiftX(kappa[:,:,:,0], 1,0)+shiftX(kappa[:,:,:,3], 0,1));
        multK[0] = 0.25*(shiftX(kappa[:,:,:,1], 1,0)+shiftX(kappa[:,:,:,2], 0,1))
        multK[1] = shiftX(kappa[:,:,:,3], 0,1);
        multK[2] = - 0.25*(shiftX(kappa[:,:,:,1], -1,0)+shiftX(kappa[:,:,:,2], 0,+1))
        multK[3] = shiftX(kappa[:,:,:,0],1,0);
        multK[5] = kappa[:,:,:,0];
        multK[6] = - 0.25*(shiftX(kappa[:,:,:,1], 1,0)+shiftX(kappa[:,:,:,2], 0,-1))
        multK[7] = kappa[:,:,:,3];
        multK[8] = + 0.25*(shiftX(kappa[:,:,:,1], -1,0)+shiftX(kappa[:,:,:,2], 0,-1))
        for j in range(-1, 2):
            for i in range(-1, 2):
                x_out = x_out + tf.multiply(tf.reshape(multK[count_k], [bSize, N, N, 1]),
                                            shiftX(x_update, i, j));
                count_k = count_k + 1;
    elif str_Knature=='scalar':
        DiscretisationFactor = tf.constant(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), dtype = tf.float32);
        i = 0;
        j = 0;
        x_out = x_out + tf.multiply(DiscretisationFactor[int(1+i),int(1+j)] , tf.multiply(tf.reshape(kappa[:, :, :, count_k], [bSize, N, N, 1]),
                                    shiftX(x_update, i, j)));

        j = 0;
        for i in [-1, 1]:
            x_out = x_out + tf.multiply(DiscretisationFactor[int(1 + i), int(1 + j)],
                                        tf.multiply(tf.reshape(kappa[:, :, :, count_k], [bSize, N, N, 1]),
                                                    shiftX(x_update, i, j)));
        i = 0;
        for j in [-1, 1]:
            x_out = x_out + tf.multiply(DiscretisationFactor[int(1 + i), int(1 + j)],
                                        tf.multiply(tf.reshape(kappa[:, :, :, count_k], [bSize, N, N, 1]),
                                                    shiftX(x_update, i, j)));

    x_out_mult = tf.multiply(dt, x_out);
    if order == 1:
        return x_out_mult
    else:
        return x_out_mult + assembleXupdate2D(x_out_mult, kappa, dt, order - 1)


def shiftX(xupdate, i, j):
    ## shifts xupdate in order to be multiplied fo kappa in a second time
    x_shifted = tf.pad(tf.reshape(xupdate, [bSize, N, N, 1]), [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC');
    x_shifted = tf.manip.roll(x_shifted, [i, j], axis=[1, 2]);
    x_out = tf.reshape(x_shifted[:, 1:N + 1, 1:N + 1, :], [bSize, N, N, 1]);
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
    maxIter = int(32000)
    print('--------------------> DiffNet Init <--------------------')
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    dataDbar = [None] * len(dataSetTest);
    for i_datasets in range(0, len(dataSetTest)):
        dataDbar[i_datasets] = loadDiff.read_data_sets(dataSetTrain, dataSetTest[i_datasets]);
    imSize = dataDbar[0].train.true.shape

    # Create the model
    imag = tf.placeholder(tf.float32, [bSize, imSize[1], imSize[2], chan])
    true = tf.placeholder(tf.float32, [bSize, imSize[1], imSize[2], 1])
    x_update = tf.zeros_like(imag);
    with tf.name_scope('DiffNet'):
        x_update = (tf.reshape(imag, [bSize, N, N, chan]))
        iii = 0;
        for iii in range(iterMain):
            x_update, kappaEst1, Kappa1 = diffLayer(x_update, bSize, N, 0 + 100 * iii)
            x_update, kappaEst2, Kappa2 = diffLayer(x_update, bSize, N, 1 + 100 * iii)
            x_update, kappaEst3, Kappa3 = diffLayer(x_update, bSize, N, 2 + 100 * iii)
            x_update, kappaEst4, Kappa4 = diffLayer(x_update, bSize, N, 3 + 100 * iii)
            x_update, kappaEst5, Kappa5 = diffLayer(x_update, bSize, N, 4 + 100 * iii)

        x_update = tf.reshape(x_update, [bSize, N, N, chan])
        x_sum = x_update[:, :, :, 0]
        for ccc in range(chan - 1):
            x_sum = x_sum + x_update[:, :, :, ccc + 1]
        x_sum = tf.reshape(x_sum, [bSize, N, N, 1]) / chan
        # y_diff = (tf.nn.relu(tf.nn.tanh(x_sum)));
        b_outo = bias_variable([1])
        y_diff = tf.reshape((tf.nn.relu(x_sum + b_outo)), [bSize, N, N, 1]);
    saver = tf.train.Saver()

    with tf.name_scope('optimizer'):
        loss = tf.norm(tf.subtract(tf.nn.relu(true), y_diff)) / float(bSize)
        rel_loss = tf.norm(tf.subtract(tf.nn.relu(true), y_diff)) / tf.norm(tf.nn.relu(true))
        ssim = tf.reduce_mean(tf.image.ssim(tf.nn.relu(true), y_diff, max_val=1.0))
        all_weights = tf.trainable_variables()
        regularizer = 0.000 * tf.add_n([tf.nn.l2_loss(v_weights) for v_weights in all_weights]);
        # regularizer = tf.nn.l2_loss(
        #         tf.image.total_variation(tf.reshape(y_diff,[bSize,N,N,1])) / (bSize* N*N));

        # rel_losst = rel_loss + regularizer;
        #     added_loss = -tf.scalar_mul(100.0,tf.minimum( tf.subtract(tf.norm(y_diff),10.0),0.0))
        #    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('rel_loss', rel_loss)
        tf.summary.scalar('psnr', psnr(y_diff, true))
        tf.summary.scalar('ssim', ssim)
        tf.summary.image('diffused', tf.reshape(imag[0, :, :, 0], [1, imSize[1], imSize[2], 1]))
        tf.summary.image('result', tf.reshape(y_diff[0], [1, imSize[1], imSize[2], 1]))
        # tf.summary.image('kappaEst1', tf.reshape(kappaEst1[1, :, :, 0], [1, imSize[1], imSize[2], 1]))
        # tf.summary.image('kappaEst2', tf.reshape(kappaEst2[1, :, :, 0], [1, imSize[1], imSize[2], 1]))
        # tf.summary.image('x_channel1', tf.nn.relu(tf.reshape(x_update[1, :, :, 0], [1, imSize[1], imSize[2], 1])))
        # tf.summary.image('x_channel2', tf.nn.relu(tf.reshape(x_update[1,:,:,1],[1, imSize[1], imSize[2], 1]) ))
        # tf.summary.image('x_channel3', tf.nn.relu(tf.reshape(x_update[1,:,:,2],[1, imSize[1], imSize[2], 1]) ))
        # tf.summary.image('x_channel4', tf.nn.relu(tf.reshape(x_update[1,:,:,3],[1, imSize[1], imSize[2], 1]) ))
        #    tf.summary.image('residual', (true - primal_result_pos)[..., zBatch // 2: zBatch // 2 + 1])
        tf.summary.image('true', tf.reshape(true[0], [1, imSize[1], imSize[2], 1]))
        #        tf.summary.image('imag', tf.reshape(imag[1,:,:,0:4],[1, imSize[1], imSize[2], 4]) )
        tf.summary.image('diff', tf.reshape(true[0] - y_diff[0], [1, imSize[1], imSize[2], 1]))
        expName = [None] * len(fileOutName)
        for i_expName in range(0, len(fileOutName)):
            expName[i_expName] = 'DiffNet_' + fileOutName[i_expName];

        test_summary_writer, train_summary_writer = util.summary_writers(name, expName, cleanup=False)

        merged_summary = tf.summary.merge_all()
    with tf.name_scope('training'):
        learningRate = tf.constant(7e-3)
        train_step = tf.train.AdamOptimizer(learningRate).minimize(rel_loss + regularizer)
        # train_step = tf.train.AdamOptimizer(learningRate).minimize(- tf.log(ssim + 1) + regularizer)

    sess.run(tf.global_variables_initializer())

    feed_test = {imag: dataDbar[0].test.images[0:bSize],
                 true: dataDbar[0].test.true[0:bSize]}

    lVal = 0.2e-4
    startIt = 0
    error_val_old= 1.2;
    for i in range(maxIter):

        batch = dataDbar[0].train.next_batch(bSize)
        #      test1 = batch[0]
        #      test2 = batch[1]
        feed_train = {imag: batch[0], true: batch[1], learningRate: lVal}

        _, merged_summary_result_train, loss_result_train = sess.run([train_step, merged_summary, rel_loss],
                                                                     feed_dict=feed_train)

        #        if i % 500 == 0:
        #            lVal = lVal*1.5

        if i % 20 == 0:
            # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            tBeg = randint(0, tRand)
            tEnd = tBeg + bSize
            it = i + startIt

            for i_test in range(0, len(dataDbar)):
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

            if i > 3000:
                # check mean validation error
                i_dataset = 1;
                array_val = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
                for i_val in range(0, np.shape(dataDbar[i_dataset].test.images)[0] - bSize+1):
                    feed_test = {imag: dataDbar[i_dataset].test.images[i_val:i_val + bSize],
                                 true: dataDbar[i_dataset].test.true[i_val:i_val + bSize]}
                    loss_val = sess.run([rel_loss], feed_dict=feed_test)
                    array_val[i_val] = np.sum(loss_val[-1]);
                error_val = np.nanmean(array_val);
                if error_val < error_val_old:
                    error_val_old = error_val
                    print('NEW CHAMPION SAVING+%f'%error_val)
                    # print(np.shape(dataDbar[0].test.images)[0])
                    saved_path = saver.save(sess, MatOutName[0][0:-4] + 'best_binaryInput')

                        # print(np.shape(dataDbar[0].test.images)[0])
    saver.restore(sess, saved_path);

    # run for all samples
    print(np.shape(dataDbar[0].test.images)[0])
    for i_dataset in range(0, len(dataDbar)):
        result = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        outK1 = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        outK2 = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        outK3 = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        outK4 = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        outK5 = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])
        result = [None] * (np.shape(dataDbar[i_dataset].test.images)[0])

        for i_final in range(0, np.shape(dataDbar[i_dataset].test.images)[0] - bSize +1 ):
            print(i_final)
            feed_test = {imag: dataDbar[i_dataset].test.images[i_final:i_final + bSize],
                         true: dataDbar[i_dataset].test.true[i_final:i_final + bSize]}
            test_result, K1,K2,K3,K4,K5 = sess.run([y_diff, kappaEst1,kappaEst2,kappaEst3,kappaEst4,kappaEst5], feed_dict=feed_test)

            result[i_final:i_final + bSize] = test_result[0:];
            outK1[i_final:i_final + bSize] = K1[0:];
            outK2[i_final:i_final + bSize] = K2[0:];
            outK3[i_final:i_final + bSize] = K3[0:];
            outK4[i_final:i_final + bSize] = K4[0:];
            outK5[i_final:i_final + bSize] = K5[0:];
        dict_sio = {
            'Input': dataDbar[i_dataset].test.images[:, :, :, :],
            'True': dataDbar[i_dataset].test.true[:, :, :, :],
            'Result': result,
            'k1': outK1,
            'k2': outK2,
            'k3': outK3,
            'k4': outK4,
            'k5': outK5,
        }
        # print(MatOutName[i_dataset]);
        sio.savemat(MatOutName[i_dataset], dict_sio)
        print("Model saved in file: %s" % MatOutName[i_dataset])
    save_path = saver.save(sess, filePath)

    # get gamma and results

    print('Result Saved')

    sess.close();
    #   tf.reset_54trg fgult_graph()

    print('--------------------> DONE <--------------------')

'''
folder_tensorboard = '';
folder_data = '../processed4python/multiGamma/2020remake/transmission/EXP/CBH/second_session/';
commonName_train = folder_data + 'EXP2_transmission_phantom';
commonName_test = folder_data + 'EXP2_transmission_phantom';
specName_train = ['2','1','3','4','5','_session2_merged','_session12_phantoms' , '_session12let_whole', '_session12_merged2345','_session12_merged1345','_session12_merged1245','_session12_merged1235']
specName_test =  [['2'],['1'],['3'],['4'],['5'],['_session2_merged'],
                  ['_session12_phantoms'], ['_session12let_whole'],['_session12_merged2345','1'],['_session12_merged1345','2'],['_session12_merged1245','3'],['_session12_merged1235','4']];#




#folder_data = '../processed4python/multiGamma/2020remake/transmission/EXP/CBH/first_session/';
#commonName_train = folder_data + 'EXP1_transmission_';
#commonName_test = folder_data + 'EXP1_transmission_';
#specName_train = ['milk_conc1let','milk_conc1','milk_conc2let', 'phantom1let'];
#specName_test =  [['milk_conc1let'],['milk_conc1'],['milk_conc2let'], ['phantom1let'] ];


#folder_tensorboard = '';
#folder_data = '../processed4python/multiGamma/2020remake/transmission/EXP/GROM/';
#commonName_train = folder_data + 'EXP_GROM_';
#commonName_test = folder_data + 'EXP_GROM_';
#specName_train = ['MILK1', 'MILK2','MILK3', 'GELMILK1','TALK1', 'TALK2','TALK3', 'MIX_TALK'];
#specName_test =  [['MILK1'], ['MILK2'],['MILK3'], ['GELMILK1'],['TALK1'], ['TALK2'],['TALK3'],
#                  ['MIX_TALK','MILK1', 'MILK2','MILK3', 'GELMILK1','TALK1', 'TALK2','TALK3', 'MIX_TALK']];
#specName_train = ['MIX_TALK'];
#specName_test =  [ ['MIX_TALK','MILK1', 'MILK2','MILK3', 'GELMILK1','TALK1', 'TALK2','TALK3']];


folder_tensorboard = '';
folder_data = '../processed4python/multiGamma/2020remake/transmission/MC/';
commonName_train = folder_data + 'DOCM4DiffNet_transmission_liqMus';
commonName_test = folder_data + 'DOCM4DiffNet_transmission_liqMus';
specName_train = ['1','2','_merged']
specName_test =  [['1','2','_merged'],['2','1','_merged'],['_merged','2','1']];

folder_tensorboard = '';
folder_data = '../processed4python/multiGamma/2020remake/transmission/PDE/FEM_PDE_avg_sca10/';
commonName_train = folder_data + 'OriginalDiff_';
commonName_test = folder_data + 'OriginalDiff_';
specName_train = ['nohete','1','2','3','4','5','ex_1_combOf_4', 'ex_2_combOf_4', 'allRandom']
specName_test =  [['nohete'],['1'],['2'],['3'], ['4'], ['5'],
                  ['ex_1_combOf_4'],['ex_2_combOf_4'],
                  ['allRandom','nohete','1','2','3','4','5','ex_1_combOf_4', 'ex_2_combOf_4']
                  ];


netPath = 'netData/test_DOCM.ckpt';
train_num = np.multiply([150 - bSize ,150 - bSize ,150 - bSize ,150 - bSize ,150 - bSize ,150 - bSize ,150 - bSize ,150 - bSize,
                         150 - bSize,
                         150 - bSize,
                         150 - bSize,
                         150 - bSize], 1)
saveMatCommon = 'RESULTS/'+'DOCM_orig_sepTest/';#implicit'
'''
folder_tensorboard = '';
folder_data = '/home/gdisciac/CBH/';

commonName_train = folder_data + 'EXP2021_';
commonName_test = folder_data + 'EXP2021_';
'''
specName_train = ['phantom1_5mm_1FT_phantom2_5mm_1FT_moving_normPeak',
                  'phantom1_5mm_1FT_moving_normPeak',
                  'phantom1_5mm_1FT_phantom2_5mm_1FT_fixed_normPeak',
                  'phantom1_5mm_1FT_fixed_normPeak',
                  'phantom1_5mm_2FT_phantom2_5mm_1FT_moving_normPeak',
                  'phantom1_5mm_2FT_moving_normPeak',
                  'phantom1_5mm_2FT_phantom2_5mm_1FT_fixed_normPeak',
                  'phantom1_5mm_2FT_fixed_normPeak'
                  ]#,'fixed_normPeak']
specName_test =  [['phantom1_5mm_1FT_phantom2_5mm_1FT_moving_normPeak','phantom1_5mm_1FT_phantom2_5mm_1FT_moving_normPeak_v'],
                  ['phantom1_5mm_1FT_moving_normPeak','phantom1_5mm_1FT_moving_normPeak_v'],
                  ['phantom1_5mm_1FT_phantom2_5mm_1FT_fixed_normPeak','phantom1_5mm_1FT_phantom2_5mm_1FT_fixed_normPeak_v'],
                  ['phantom1_5mm_1FT_fixed_normPeak','phantom1_5mm_1FT_fixed_normPeak_v'],
                  ['phantom1_5mm_2FT_phantom2_5mm_1FT_moving_normPeak','phantom1_5mm_2FT_phantom2_5mm_1FT_moving_normPeak_v'],
                  ['phantom1_5mm_2FT_moving_normPeak','phantom1_5mm_2FT_moving_normPeak_v'],
                  ['phantom1_5mm_2FT_phantom2_5mm_1FT_fixed_normPeak','phantom1_5mm_2FT_phantom2_5mm_1FT_fixed_normPeak_v'],
                  ['phantom1_5mm_2FT_fixed_normPeak', 'phantom1_5mm_2FT_fixed_normPeak_v']
                  ];
'''
specName_train = [ sys.argv[1] ]
specName_test = [ [sys.argv[1], sys.argv[1]+'_v']];
netPath = 'netData/test_DOCM.ckpt';
train_num = np.multiply([200 - bSize ,200 - bSize ,200 - bSize ,200 - bSize ,200 - bSize ,200 - bSize ,200 - bSize ,200 - bSize,
                         200 - bSize,
                         200 - bSize,
                         200 - bSize,
                         200 - bSize], 1)
saveMatCommon = 'RESULTS/'+'DOCM_orig_sepTest/';#implicit'
logNameCommon = folder_tensorboard +'orig_explKappaScalar_mc';


# outL = main(netPath,logName,dataSetTrain,dataSetTest, 1050,savematName)

for j_proc in range(0, len(specName_train)):
    dataSetTest = [None] * len(specName_test[j_proc]);
    logName = [None] * len(specName_test[j_proc]);
    savematName = [None] * len(specName_test[j_proc]);
    dataSetTrain = commonName_train + specName_train[j_proc] + '.mat';
    for i_logname in range(0, len(specName_test[j_proc])):
        dataSetTest[i_logname] = commonName_test + specName_test[j_proc][i_logname] + '_t.mat';
        logName[i_logname] = logNameCommon + '_train_' + specName_train[j_proc] + '_test_' + specName_test[j_proc][
            i_logname];
        savematName[i_logname] = util.default_tensorboard_dir(name) + 'Matlab' +'_'+logNameCommon +'_train_' + specName_train[
            j_proc] + '_test_' + specName_test[j_proc][i_logname] + '.mat';
    print(logName)
    # Lout,output =main(netPath,logName,dataSetTrain,dataSetTest, train_num[j_proc], savematName)
    p = multiprocessing.Process(target=main,
                                args=(netPath, logName, dataSetTrain, dataSetTest, train_num[j_proc], savematName));

    p.start()
    p.join()
    p.terminate()
