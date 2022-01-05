#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:52:42 2019

@author: gdisciac
"""
## U-net for priors from optical ad us data


import tensorflow as tf
import sys

sys.path.append("..")
import Load_DiffNet_Inv as loadDiff
import numpy as np
import utilTemp_nonlinInv_mult as util
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0';
#import scipy.sparse as ssp
from scipy import sparse as ssp
from scipy.sparse import linalg as splin
from scipy.sparse import csr_matrix as spcrs
import scipy
import scipy.io as sio  # %ok- 1
from random import randint
# from numba import cuda
import matplotlib.pyplot as plt
import multiprocessing
import h5py
# import time

print('pausing')
# time.sleep(3600*2)
print('resuming')
# import matplotlib.pyplot as plt

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

bSize = int(1)
chan = int(1)

wchan = int(64);
Nx = int(32);
Ny = int(29)
Nz = int(16)
Nt = int(10);
nsource = int(8);
ndetector = int(8);
nfreq = 20;
Nm = int(nsource * (ndetector-1))
size_domain = Nx*Ny*Nz;


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
    x =  np.sqrt(2/(shape[0]*shape[1]*shape[2]*(shape[3]+shape[4])));
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

def u_net_weight_down(laynum, in_channels, out_channels):
    W = [None]*3;
    W[0] = weight_variable([3, 3, 3, in_channels, out_channels], str(laynum) + '1'+str(randint(0, 100000)))
    W[1] = weight_variable([3, 3, 3, out_channels, out_channels], str(laynum) + '2'+str(randint(0, 100000)))
    W[2] = weight_variable([3, 3, 3, out_channels, out_channels], str(laynum)+'3'+str(randint(0, 100000)))
    return W

def u_net_weight_up(laynum, in_channels, out_channels):

    W = [None]*3;
    #W[0] = weight_variable([3, 3, 3, out_channels, in_channels], str(laynum) + '1')
    W[0] = weight_variable([3, 3, 3,in_channels ,out_channels], str(laynum) + '1'+str(randint(0, 100000)))
    W[1] = weight_variable([3, 3, 3, out_channels, out_channels], str(laynum) + '2'+str(randint(0, 100000)))
    W[2] = weight_variable([3, 3, 3, out_channels, out_channels], str(laynum) + '3'+str(randint(0, 100000)))
    return W

def u_net_step_down(x, W):

    b = [None] * 3;
    b[0] = bias_variable([W[0].get_shape()[4]])
    b[1] = bias_variable([W[1].get_shape()[4]])
    b[2] = bias_variable([W[2].get_shape()[4]])
    print(W[0]);
    print(W[1])
    print(W[2])
    print(x)
    l1 = tf.nn.elu(conv3d(x,W[0])) + b[0];
    l2 = tf.nn.elu(conv3d(l1,W[1])) + b[1];
    l3 = conv3d(l2, W[2]);
    out = tf.nn.max_pool3d( tf.nn.elu(tf.add(l3, l1)), [1,5,5,5,1], [1,1,1,1,1], padding='SAME') + b[2];
    return out;


def u_net_step_up(x, W, output_shape):
    b = [None] * 3;
    #b[0] = bias_variable([W[0].get_shape()[3]])
    b[0] = bias_variable([W[0].get_shape()[4]])
    b[1] = bias_variable([W[1].get_shape()[4]])
    b[2] = bias_variable([W[2].get_shape()[4]])
    #l1 = tf.nn.relu(d_conv3d(x, W[0], output_shape)) + b[0];
    l1 = tf.nn.elu(conv3d(x, W[0])) + b[0];
    l2 = tf.nn.elu(conv3d(l1, W[1]))+ b[1];
    l3 = conv3d(l2, W[2]);
    #out = tf.nn.elu(tf.add(l3, l1))+ b[2];
    out = (tf.add(l3 +b[2], l1));
    #out = tf.nn.max_pool3d(tf.nn.elu(tf.add(l3, l1)), [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME') +b[2]
    return out

def UNETd(inValue, chan_net = chan):
    W0 = weight_variable([3,3,3,chan_net,int(wchan/4)], 'UNETd'+str(randint(0, 100000)));
    l0 = conv3d(inValue, W0);
    W1_d = u_net_weight_down(randint(0, 100000), int(int(int(wchan)/4)), int(int(wchan)/4))
    b1_d = bias_variable([int(int(wchan)/4)]);
    l1_d = u_net_step_down(l0, W1_d) + b1_d #32
    W2_d = u_net_weight_down(randint(0, 100000), int(int(wchan)/4), int(int(wchan)/2))
    b2_d = bias_variable([int(int(wchan)/2)]);
    l2_d = u_net_step_down(l1_d, W2_d) + b2_d#64
    W3_d = u_net_weight_down(randint(0, 100000), int(int(wchan)/2), int(wchan))
    b3_d = bias_variable([int(wchan)]);
    l3_d = u_net_step_down(l2_d, W3_d) + b3_d #128
    W4_d = u_net_weight_down( randint(0, 100000), int(wchan), int(wchan))
    b4_d = bias_variable([int(wchan)]);
    l4_d = u_net_step_down(l3_d, W4_d) + b4_d #128

    return tf.nn.elu(l0),tf.nn.elu(l1_d), tf.nn.elu(l2_d), tf.nn.elu(l3_d),tf.nn.elu(l4_d)
def UNET(inValue0, inValue1, inValue2, outchan=1):
    # down
    # W0
    l0 = 0;
    _, a0, b0, c0, d0 = UNETd(inValue0)
    _, a1, b1, c1, d1 = UNETd(inValue1)
    _, a2, b2, c2, d2 = UNETd(inValue2)

    wout4 = weight_variable([3, 3, 3, 3 * int(wchan), int(wchan)], 'zw101010'+str(randint(0, 100000)));
    l4_d = conv3d(tf.concat([ d0, d1, d2], axis = 4), wout4);
    wout3 = weight_variable([3,3,3,3*int(wchan),int(wchan)], 'zw10101'+str(randint(0, 100000)));
    l3_d = conv3d(tf.concat([ c0, c1, c2], axis = 4), wout3);
    wout2 = weight_variable([3,3,3,3*int(int(wchan)/2),int(int(wchan)/2)], 'zw10102'+str(randint(0, 100000)));
    l2_d = conv3d(tf.concat([ b0, b1,b2], axis = 4), wout2);
    wout1 = weight_variable([3,3,3,3*int(int(wchan)/4),int(int(wchan)/4)], 'zw10103'+str(randint(0, 100000)));
    l1_d = conv3d(tf.concat([ a0, a1,a2], axis = 4), wout1);#a2#+a1+a2#tf.concat([ a0, a1, a2], axis = 4) ;

############################
    ''''
    wout4 = weight_variable([3, 3, 3, 128, 128], 'zw101010');
    l4_d = conv3d( d2, wout4);
    wout3 = weight_variable([3,3,3,128,128], 'zw10101');
    l3_d = conv3d(c2, wout3);
    wout2 = weight_variable([3,3,3,64,64], 'zw10102');
    l2_d = conv3d( b2, wout2);
    wout1 = weight_variable([3,3,3,32,32], 'zw10103');
    l1_d = conv3d( a2, wout1);#a2#+a1+a2#tf.concat([ a0, a1, a2], axis = 4) ;
    '''
############################
    # up

    W1_u = u_net_weight_up(5, int(wchan), int(wchan))
    b1_u = bias_variable([int(wchan)]);
    l1_u = tf.add(u_net_step_up(l4_d, W1_u, [bSize, Nx, Ny, Nz, int(wchan)]), l4_d) + b1_u ;#128
    W2_u = u_net_weight_up(6, int(wchan), int(wchan))
    b2_u = bias_variable([int(wchan)]);
    l2_u = tf.add(u_net_step_up(l1_u, W2_u, [bSize, Nx, Ny, Nz, int(wchan)]),l3_d) + b2_u ; #128
    W3_u = u_net_weight_up(7, int(wchan), int(int(wchan)/2))
    b3_u = bias_variable([int(int(wchan)/2)])
    l3_u = tf.add(u_net_step_up(l2_u, W3_u, [bSize, Nx, Ny, Nz, int(int(wchan)/2)]), l2_d) + b3_u ;#64
    W4_u = u_net_weight_up(8, int(int(wchan)/2), int(int(wchan)/4))
    b4_u = bias_variable([int(int(wchan)/4)]);
    l4_u = tf.add(u_net_step_up(l3_u, W4_u, [bSize, Nx, Ny, Nz, int(int(wchan)/4)]),l1_d) + b4_u ;
    Wout_u = u_net_weight_up(9, int(int(wchan)/4), int(int(wchan)/4))
    bout_u = bias_variable([int(int(wchan)/4)]);
    out = tf.add(u_net_step_up(l4_u, Wout_u, [bSize, Nx, Ny, Nz, int(int(wchan)/4)]), l0) + bout_u;
    wout = weight_variable([3,3,3,int(int(wchan)/4),outchan], '101'+str(randint(0, 100000)));
    out = conv3d(out, wout);
    wout_o = weight_variable([7, 7, 7, outchan+3*chan, outchan], 'o10100011'+str(randint(0, 100000)));
    return conv3d( tf.concat([out, inValue0, inValue1, inValue2], axis = 4), wout_o)
#    wout_o = weight_variable([3, 3, 3, 2, 1], 'o10100011');
#    return conv3d( tf.concat([out, inValue2], axis = 4), wout_o)

def getKappa(inVal, shape, varName):
    kappa = tf.constant(inVal, shape=shape)
    return tf.Variable(kappa, name=varName)


def assembleJ(J, prior_a, prior_s, prior_as, prior_sa):
    alpha = bias_variable([1])
    prior_a = alpha* tf.concat([prior_a, prior_as], axis = 2)
    prior_s = alpha* tf.concat([ prior_sa, prior_s],axis = 2)
    Jp = tf.concat([J, prior_a, prior_s], axis = 1);
    return Jp


def isoLap():
    Onesx = np.ones([Nx])
    Onesy = np.ones([Ny])
    Onesz = np.ones([Nz])
    D1dz = np.diag(Onesz, k = 0) + np.diag(- Onesz[0:-1], k = -1)
    D1dy = np.diag(Onesy, k=0) + np.diag(- Onesy[0:-1], k=-1)
    D1dx = np.diag(Onesx, k=0) + np.diag(- Onesx[0:-1], k=-1)

    Eyex = np.eye(Nx)
    Eyey = np.eye(Ny)
    Eyez = np.eye(Nz)
    L0z = np.kron(np.kron(D1dz, Eyey), Eyex)
    L0y = np.kron(np.kron(Eyez,D1dy), Eyex)
    L0x = np.kron(Eyez, np.kron(Eyey, D1dx))
    return L0x, L0y, L0z,tf.constant(L0x, dtype=tf.float32),tf.constant(L0y, dtype=tf.float32), tf.constant(L0z, dtype=tf.float32)


def tf3Dgradient(p):
    # x = [batch, length, width, depth, chan]
    p0 = tf.pad(p, [[0,0],[1,1],[1,1],[1,1],[0,0]]);
    gx = tf.manip.roll(p0, shift=1, axis = 1) - p0;
    gy = tf.manip.roll(p0, shift=1, axis = 2) - p0;
    gz = tf.manip.roll(p0, shift=1, axis = 3) - p0;
    return gx[:,1:-1,1:-1,1:-1,:], gy[:,1:-1,1:-1,1:-1,:],gz[:,1:-1,1:-1,1:-1,:]

def getLaplacian(p, str_behaviour = 'fromSingle'):

    _, _, _, L0x, L0y, L0z = isoLap();
    if str_behaviour == 'fromSingle':
        px,py,pz = tf3Dgradient(p)
        gg = tf.square(px)+tf.square(py)+tf.square(pz)
        kap = tf.sqrt(tf.exp(tf.divide(tf.negative(gg), tf.reduce_max(gg, axis=[1,2,3,4]) )))
        kap = tf.transpose(kap, [0,3,2,1,4]);
        kap = tf.linalg.diag(tf.reshape(kap, [bSize, -1]));

        Lx = tf.multiply(kap, tf.expand_dims(L0x, axis=0));
        Ly = tf.multiply(kap, tf.expand_dims(L0y, axis=0));
        Lz = tf.multiply(kap, tf.expand_dims(L0z, axis=0));

        return tf.concat([Lx,Ly,Lz], axis=1);

    elif str_behaviour == 'fromDiags':
        p = tf.transpose(p, [0, 3, 2, 1, 4]);
        kap = [None]*6;
        for idiag in range(0,6):
            kap[idiag] = tf.linalg.diag(tf.reshape(p[:,:,:,:,idiag, bSize,-1]));

        Lx1 = tf.multiply(kap[0], 0.5 *(tf.abs(L0x) + L0x));
        Lxm1 = tf.multiply(kap[1], 0.5 *(- tf.abs(L0x) + L0x));
        Ly1 = tf.multiply(kap[2], 0.5 *(tf.abs(L0y) + L0y));
        Lym1 = tf.multiply(kap[3], 0.5 *(- tf.abs(L0y) + L0y));
        Lz1 = tf.multiply(kap[4], 0.5 *(tf.abs(L0z) + L0z));
        Lzm1 = tf.multiply(kap[5], 0.5 *(- tf.abs(L0z) + L0z));
        Lx = Lx1 +Lxm1;
        Ly = Ly1 + Lym1;
        Lz = Lz1 + Lzm1;

    return tf.concat([Lx,Ly,Lz], axis=1);

def py_getLap(p):
    px, py, pz = np.gradient(np.float32(p), axis = [1,2,3])
    gg = np.square(px) + np.square(py) + np.square(pz)
    kap = np.sqrt(np.exp(- tf.divide(gg, np.amax(gg, axis=(1,2,3)))))
    kap = np.transpose(kap, (0, 3, 2, 1));
    kap = np.diag(np.reshape(kap, [bSize, -1]));
    L0x, L0y, L0z,_,_,_ = isoLap();
    Lx = np.multiply(kap, np.expand_dims(L0x, axis=0));
    Ly = np.multiply(kap, np.expand_dims(L0y, axis=0));
    Lz = np.multiply(kap, np.expand_dims(L0z, axis=0));
    return np.concatenate([Lx,Ly,Lz], axis = 1)



def overlap(true, guess):
    npweight =np.tile( 1+np.abs(np.linspace( [-0.5*Ny],[0.5*Ny], Ny)),[bSize, Nx, 1, Nz])
    npweight = np.ones_like(npweight);
    weight = tf.constant(npweight, dtype = tf.float32);
    weight = tf.reshape(weight,[bSize*Nx*Ny*Nz]);
    true = tf.multiply(weight,tf.reshape(tf.nn.relu(true), [bSize * Nx * Ny * Nz]));
    guess = tf.multiply(weight,tf.reshape(tf.nn.relu(guess), [bSize * Nx * Ny * Nz]));
    fmax = tf.reduce_sum(tf.maximum(true, guess))
    fmin = tf.reduce_sum(tf.minimum(true,guess))
    TC = tf.divide(fmin,fmax);
    return 1 - TC#tf.divide(2*TC,TC+1)

def main(filePath, fileOutName, dataSetTrain, dataSetTest, tRand, MatOutName):
    iterMain = int(1)
    maxIter = int(18000)

    print('--------------------> DiffNet Init <--------------------')
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    dataDbar = [None]*len(dataSetTest);
    for i_datasets in range(0, len(dataSetTest) ):
        dataDbar[i_datasets] = loadDiff.read_data_sets(dataSetTrain, dataSetTest[i_datasets], '');
    muarecon0 = tf.placeholder(tf.float32, [bSize, Nx,Ny,Nz, chan])
    musrecon0 = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    truea = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    trues = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    s = tf.placeholder(tf.float32, [bSize, nfreq,nfreq])
    U = tf.placeholder(tf.float32, [bSize, Nt*Nm, nfreq])
    V = tf.placeholder(tf.float32, [bSize,Nx*Ny*Nz*2,nfreq])
    data = tf.placeholder(tf.float32, [bSize,Nt*Nm,1]);
    prior2D = tf.placeholder(tf.float32, [bSize, Nx,Nz])
    prior3D = tf.placeholder(tf.float32, [bSize, Nx,Ny,Nz,1]);


    #prior3D = tf.expand_dims(prior3D0, axis = 4)
    #prior2D = tf.placeholder(tf.float32, [bSize, Nx,Nz]);

    with tf.name_scope('Net'):
        #tf.tile( tf.eye(Nx*Nz, num_columns=Nz*Nx), [Ny,1] )
        #b0 = bias_variable([1])
        W2D3D = ((1)/(Nx*Nz))* tf.get_variable('W2D3D',shape=[Nx*Ny*Nz,Nz*Nx], initializer=tf.contrib.layers.xavier_initializer())
        prior0 =(tf.matmul(tf.tile(tf.expand_dims( W2D3D, axis = 0),[bSize, 1,1]),
                                      tf.reshape(prior2D, [bSize,-1,1]),
                                    ))
        prior = tf.transpose(tf.reshape(prior0 , [bSize, Nx,Nz,Ny,1]), [0,1,3,2,4])

        bb = bias_variable([1])

        for i_rec in range(0,1):
            recon_prior = (UNET(muarecon0, musrecon0, (prior))+ bb)

        # compute optical reconstruction
        out_mua = recon_prior;
        out_mus = recon_prior;
    with tf.name_scope('optimizer'):
        rel_loss_p = tf.norm(tf.subtract((recon_prior), prior3D)) / tf.norm(prior3D)
        rel_loss_a = tf.norm(tf.subtract(tf.nn.relu(out_mua), prior3D)) / tf.norm(prior3D)
        rel_loss_s = tf.norm(tf.subtract(tf.nn.relu(out_mus), prior3D)) / tf.norm(prior3D)
        rel_loss = 0 * rel_loss_a + 0* rel_loss_s+rel_loss_p;

    with tf.name_scope('summaries'):
        tf.summary.scalar('rel_loss_p', rel_loss_p)
        tf.summary.scalar('rel_loss_s', rel_loss_s)
        tf.summary.scalar('rel_loss_a', rel_loss_a)
        tf.summary.scalar('rel_loss', rel_loss)

        expName = [None]*len(fileOutName);

        for i_expName in range(0, len(fileOutName)):
            expName[i_expName] = 'DiffNet_' + fileOutName[i_expName];

        test_summary_writer, train_summary_writer = util.summary_writers(name, expName, cleanup=False)


        merged_summary = tf.summary.merge_all()

        with tf.name_scope('training'):
            learningRate = tf.constant(5e-6)
            #train_step = tf.train.AdamOptimizer(learningRate).minimize(rel_loss+ 0.1*rel_loss0)
            train_step = tf.train.AdamOptimizer(learningRate).minimize(rel_loss)

    sess.run(tf.global_variables_initializer())
    lVal = 1e-4
    ''''
    feed_test = {recona: dataDbar[0].test.recona[0:bSize],
                 recons: dataDbar[0].test.recons[0:bSize],
                 prior3D0: dataDbar[0].test.prior[0:bSize],
                 prior2D: dataDbar[0].test.priorslice[0:bSize]}
    '''
    startIt = 0
    for i in range(maxIter):

        batch = dataDbar[0].train.next_batch(bSize)
        #      test1 = batch[0]
        #      test2 = batch[1]
        feed_train = {muarecon0: float(sys.argv[2]) * np.float32(np.array(batch['recona'])),
                      musrecon0: float(sys.argv[2]) * np.float32(np.array(batch['recons'])),
                      prior3D: np.float32(np.array(batch['prior'])),
                      prior2D: float(sys.argv[3])*np.float32(np.array(batch['priorslice'])),
                      trues:np.float32(np.array(batch['trues'])),
                      truea:np.float32(np.array(batch['truea'])),
                      s:np.float32(np.array(batch['s'])),
                      U:np.float32(np.array(batch['U'])),
                      V:np.float32(np.array(batch['V'])),
                      data:np.float32(np.array(batch['data'])),
                      learningRate: lVal}

        _, merged_summary_result_train, loss_result_train, prior_ = sess.run(
                [train_step, merged_summary, rel_loss, prior], feed_dict=feed_train)

        if i % 50 == 0: #write test
            # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            for i_test in range(0,len(dataDbar)):
                tBeg = randint(0, tRand[i_test])
                tEnd = tBeg + bSize
                it = i + startIt

                feed_test = {muarecon0: dataDbar[i_test].test.recona[tBeg:tEnd],
                             musrecon0: dataDbar[i_test].test.recons[tBeg:tEnd],
                             prior3D: dataDbar[i_test].test.prior[tBeg:tEnd],
                             prior2D: dataDbar[i_test].test.priorslice[tBeg:tEnd],
                             trues: dataDbar[i_test].test.trues[tBeg:tEnd],
                             truea: dataDbar[i_test].test.truea[tBeg:tEnd],
                             data:dataDbar[i_test].test.data[tBeg:tEnd],
                             s: dataDbar[i_test].test.s[tBeg:tEnd],
                             U: dataDbar[i_test].test.U[tBeg:tEnd],
                             V: dataDbar[i_test].test.V[tBeg:tEnd]}

                #test_accuracy, test_result = sess.run([loss, prior_mua],
                #                                      feed_dict=feed_test)

                rel_loss_res_p,rel_loss_res_a,rel_loss_res_s, merged_summary_result = sess.run([ rel_loss_p,rel_loss_a,rel_loss_s, merged_summary],
                                                                            feed_dict=feed_test)

                train_summary_writer.add_summary(merged_summary_result_train, it)
                test_summary_writer[i_test].add_summary(merged_summary_result, it)

                print('iter={}, rel.lossp={}, rel.lossa.test={}, rel.losss.train={}'.format(i,rel_loss_res_p,rel_loss_res_a,rel_loss_res_s ))

    # SAVING
    for i_dataset in range(0, len(dataDbar)):
        resulta = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])
        results = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])

        for i_final in range(0, np.shape(dataDbar[i_dataset].test.recona)[0] - bSize + 1):
            tBeg = i_final
            tEnd = i_final + bSize;
            feed_test = {muarecon0: dataDbar[i_test].test.recona[tBeg:tEnd],
                         musrecon0: dataDbar[i_test].test.recons[tBeg:tEnd],
                         prior3D: dataDbar[i_test].test.prior[tBeg:tEnd],
                         prior2D: dataDbar[i_test].test.priorslice[tBeg:tEnd],
                         trues: dataDbar[i_test].test.trues[tBeg:tEnd],
                         truea: dataDbar[i_test].test.truea[tBeg:tEnd],
                         s: dataDbar[i_test].test.s[tBeg:tEnd],
                         U: dataDbar[i_test].test.U[tBeg:tEnd],
                         V: dataDbar[i_test].test.V[tBeg:tEnd]}

            test_resulta,test_results = sess.run([out_mua,out_mus], feed_dict=feed_test)
            resulta[i_final:i_final + bSize] = test_resulta[0:];
            results[i_final:i_final + bSize] = test_results[0:];

        dict_sio = {'muarecon0': dataDbar[i_test].test.recona,
                    'musrecon0': dataDbar[i_test].test.recons,
                    'prior3D': dataDbar[i_test].test.prior,
                    'prior2D': dataDbar[i_test].test.priorslice,
                    'trues': dataDbar[i_test].test.trues,
                    'truea': dataDbar[i_test].test.truea,
                    's': dataDbar[i_test].test.s,
                    'U': dataDbar[i_test].test.U,
                    'V': dataDbar[i_test].test.V,
                    'resulta': resulta,
                    'results': results}
        # print(MatOutName[i_dataset]);
        sio.savemat(MatOutName[i_dataset], dict_sio)
        # save_path = saver.save(sess, filePath)
        print("Model saved in file: %s" % MatOutName[i_dataset])

    sess.close();
    #   tf.reset_default_graph()
#    print('--------------------> DONE <--------------------')

folder_tensorboard = '';
#folder_data = '/cs/research/medim/gdisciac/OpticalDatasetPrior/'#'../processed4python/multiGamma/original_diffnet/DOCM_reflection/';
folder_data = '/home/gdisciac/OpticalDatasetPrior/'
commonName_train = folder_data + 'PRIOR4DOT_';
commonName_test = folder_data + 'PRIOR4DOT_';
#specName_train = ['s1anisox']#norm', 'BLOB3'];#sys.argv[4]
#specName_test =  [['s1anisox']]#norm','EXPCILnorm','EXPCILnorma0','EXPCILnorms0','EXPCILa0s0'] ,['BLOB3','EXPCIL','EXPCILa0','EXPCILs0','EXPCILa0s0']]#, [ 'CUBE','BLOB' ]];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];
specName_train = [ sys.argv[4]];
specName_test =  [[sys.argv[4]]];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];

#specName_train = [ 'allRandom' ,'nohete', 'from1to4' ];
#specName_test = [['allRandom','from1to4','nohete'] ,['nohete', 'from1to4', 'allRandom'],['from1to4','nohete', 'allRandom']];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];
netPath = 'netData/test_DOCM.ckpt';
train_num = [ np.multiply( [ 200 - bSize -1, 100 - bSize -1,200,200,200],1)];
saveMatCommon = 'RESULTS/'+'DOCM_orig_sepTest/';#implicit'
logNameCommon = folder_tensorboard +'3UNET__';


for j_proc in range(0, len(specName_train)):
    dataSetTest = [None] * len(specName_test[j_proc]);
    logName = [None] * len(specName_test[j_proc]);
    savematName = [None] * len(specName_test[j_proc]);
    dataSetTrain = commonName_train + specName_train[j_proc] + '.mat';
    for i_logname in range(0, len(specName_test[j_proc])):
        dataSetTest[i_logname] = commonName_test + specName_test[j_proc][i_logname] + '_t.mat';
        logName[i_logname] = logNameCommon + '_train_' + specName_train[j_proc] + '_test_' + specName_test[j_proc][
            i_logname];
        savematName[i_logname] = util.default_tensorboard_dir('') +'Matlab'+ logNameCommon +'_train_' + specName_train[j_proc] + '_test_' + specName_test[j_proc][i_logname] + '.mat';
    print(logName)
    # Lout,output =main(netPath,logName,dataSetTrain,dataSetTest, train_num[j_proc], savematName)
    p = multiprocessing.Process(target=main,
                                args=(netPath, logName, dataSetTrain, dataSetTest, train_num[j_proc], savematName));

    p.start()
    p.join()
    p.terminate()


'''''
 python DeepPrior_GaussBlur.py 'full' 1 1 's1anisox';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisox';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisox';python DeepPrior_GaussBlur.py 'full' 1 1 's1anisoy';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoy';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoy';python DeepPrior_GaussBlur.py 'full' 1 1 's1anisoz';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoz';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisox';python DeepPrior_GaussBlur.py 'full' 1 1 's1anisoxinv';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoxinv';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoxinv';python DeepPrior_GaussBlur.py 'full' 1 1 's1anisoyinv';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoyinv';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoyinv';python DeepPrior_GaussBlur.py 'full' 1 1 's1anisozinv';python DeepPrior_GaussBlur.py 'blur' 1 0 's1anisozinv';python DeepPrior_GaussBlur.py 'slice' 0 1 's1anisozinv';
'''''