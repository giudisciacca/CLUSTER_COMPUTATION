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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import Load_DiffNet_Inv as loadDiff
import numpy as np
import utilTemp_nonlinInv_mult as util
import os
import tensorflow_probability as tfp

#os.environ['CUDA_VISIBLE_DEVICES'] = '0';
#import scipy.sparse as ssp
from scipy import sparse as ssp
import scipy.io as sio  # %ok- 1
import random
from random import randint
random.seed(10)
from scipy.sparse import linalg as ssplin
import scipy as sc
# from numba import cuda
import matplotlib.pyplot as plt
import multiprocessing
import util_unet as utun
import h5py
from tensorflow.python.framework import ops
# import time
config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
config.gpu_options.allow_growth = True

print('pausing')
# time.sleep(3600*2)
print('resuming')
# import matplotlib.pyplot as plt

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

bSize = int(1)
chan = int(1)
wchan = int(32);
spformat = 'csr'
global_behav = 'fromSingle';

sys.argv.append('full_1iter_dbg')
sys.argv.append(0)
sys.argv.append(1)
sys.argv.append('BORNsvdLarge2USonly')
#sys.argv.append('fromSingle')
sys.argv.append(1)


#global_behav = sys.argv[5];


Nx = int(32)#int(28)#;
Ny = int(29)#int(26)#
Nz = int(16)#int(14)#
Nt = int(20);
nsource = int(8);
ndetector = int(8);
nfreq = 20;
Nm = int(nsource * (ndetector-1))
size_domain = Nx*Ny*Nz;
NxUS = int(346);
NzUS = int(578);

if 'USonly' in sys.argv[4]:
    NzUS = int(578);
    NxUS = int(346)
    #NzUS = int(265);tf.get_shape(prior2Dconv)[2]
    #NxUS = int(133);

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

    _,a0, b0, c0, d0 = UNETd(inValue0)
    _,a1, b1, c1, d1 = UNETd(inValue1)
    _,a2, b2, c2, d2 = UNETd(inValue2)

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
        p = tf.transpose(p, [0, 3, 2, 1, 4]);
        px,py,pz = tf3Dgradient(p)
        p = tf.square(px)+tf.square(py)+tf.square(pz)
        kap = tf.sqrt(tf.exp(tf.divide(tf.negative(p), tf.reduce_max(p, axis=[1,2,3,4]) )))
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
            kap[idiag] = tf.linalg.diag(tf.reshape(0.5*p[:,:,:,:,idiag, bSize,-1]));

        Lx1 = np.multiply(kap[0], (np.abs(L0x) + L0x));
        Lxm1 = np.multiply(kap[1], (-np.abs(L0x) + L0x));
        Ly1 = np.multiply(kap[2], (np.abs(L0y) + L0y));
        Lym1 = np.multiply(kap[3], (-np.abs(L0y) + L0y));
        Lz1 = np.multiply(kap[4], (np.abs(L0z) + L0z));
        Lzm1 = np.multiply(kap[5], (-np.abs(L0z) + L0z));
        Lx = Lx1 + Lxm1;
        Ly = Ly1 + Lym1;
        Lz = Lz1 + Lzm1;

    return tf.concat([Lx,Ly,Lz], axis=1);
def ssp_isoLap():
    Onesx = np.ones([Nx])
    Onesy = np.ones([Ny])
    Onesz = np.ones([Nz])

    D1dz = ssp.spdiags(Onesz, 0, Nz, Nz) + ssp.spdiags(- Onesz[0:-1], -1, Nz, Nz)
    D1dy = ssp.spdiags(Onesy, 0, Ny, Ny) + ssp.spdiags(- Onesy[0:-1], -1, Ny, Ny)
    D1dx = ssp.spdiags(Onesx, 0, Nx, Nx) + ssp.spdiags(- Onesx[0:-1], -1, Nx, Nx)

    Eyex = ssp.eye(Nx)
    Eyey = ssp.eye(Ny)
    Eyez = ssp.eye(Nz)
    L0z = ssp.kron(ssp.kron(D1dz, Eyey), Eyex)
    L0y = ssp.kron(ssp.kron(Eyez,D1dy), Eyex)
    L0x = ssp.kron(Eyez, ssp.kron(Eyey, D1dx))

    return L0x.asformat(spformat),L0y.asformat(spformat),L0z.asformat(spformat)
GL0x, GL0y, GL0z= ssp_isoLap()

def py_getLap(kap):
    kap = np.diag(np.reshape(kap, [bSize, -1]));
    #L0x, L0y, L0z,_,_,_ = isoLap();
    L0x = GL0x;
    L0y = GL0y;
    L0z = GL0z;
    Lx = np.multiply(kap, np.expand_dims(L0x, axis=0));
    Ly = np.multiply(kap, np.expand_dims(L0y, axis=0));
    Lz = np.multiply(kap, np.expand_dims(L0z, axis=0));
    return np.concatenate([Lx,Ly,Lz], axis = 1)


def ssp_getLaplacian(p, str_behaviour = 'fromSingle'):

    #if np.shape(p)[0] != 1:
     #   raise Exception("Sparse implementation does not support batches of more than one sample")

    #L0x, L0y, L0z = ssp_isoLap();
    #p = np.transpose(p, [0, 3, 2, 1, 4]);
    #if str_behaviour == 'fromSingle':
        #p = np.ones([1,Nz,Ny,Nx,1]);
    #kap = p;
    ''''
    px, py, pz = np.gradient(np.float32(p), axis=[1, 2, 3])
    gg = np.square(px) + np.square(py) + np.square(pz)+0.001;
    kap = np.sqrt(np.exp(- np.divide(5*gg, np.amax(gg, axis=(1, 2, 3)))))
    '''
    kap = ssp.csr_matrix(ssp.diags(np.reshape(p, [ -1])));
    Lx = kap.dot(GL0x);
    Ly = kap.dot(GL0y);
    Lz = kap.dot(GL0z);
    ''''
    elif str_behaviour == 'fromDiags':
        kap = [None]*6;
        for idiag in range(0,6):
            kap[idiag] = ssp.spdiags(np.reshape(0.5*p[:,:,:,:,idiag], [-1]), 0, Nx*Ny*Nz, Nx*Ny*Nz);

        Lx1 = np.multiply(kap[0],(np.abs(L0x) + L0x));
        Lxm1 = np.multiply(kap[1],(-np.abs(L0x) + L0x));
        Ly1 = np.multiply(kap[2],(np.abs(L0y) + L0y));
        Lym1 = np.multiply(kap[3],(-np.abs(L0y) + L0y));
        Lz1 = np.multiply(kap[4],(np.abs(L0z) + L0z));
        Lzm1 = np.multiply(kap[5],(-np.abs(L0z) + L0z));
        Lx = Lx1 +Lxm1;
        Ly = Ly1 + Lym1;
        Lz = Lz1 + Lzm1;
    '''

    return ssp.vstack([Lx,Ly,Lz]).asformat(spformat);

def ssp_assembleJ(J, prior_a, prior_s, prior_as, prior_sa):
    prior_a = ssp.hstack([prior_a, prior_as]).asformat(spformat);
    prior_s = ssp.hstack([prior_sa, prior_s]).asformat(spformat);
    Jp = ssp.vstack([J, prior_a, prior_s]).asformat(spformat);
    return Jp

def ssp_solve(Jin, data, prior_tot, alpha, c_calc):
    out = np.zeros([bSize, Nx * Ny * Nz * 2], dtype=np.float32);
    if c_calc == 0:
         return out;
    Jin = np.float32(Jin);
    data = np.float32(data)
    alpha = np.float32(alpha)
    prior_tot = np.float32(prior_tot)
    datazeros = np.zeros([Nx * Ny * Nz * 3 * 2, ], dtype=np.float32);

    for i in range(0,np.shape(Jin)[0]):
        # sparsify and float32
        #J = ssp.csr_matrix(U[i,:,:]@(s[i,:,:]@V[i,:,:].T));
        J = ssp.csr_matrix(Jin[i,:,:]);
        data = np.array(np.reshape(data[i,:,:],[-1,]));
        # assemble
        datap = np.hstack([data,datazeros]);
        #if global_behav == 'fromSingle':
        L = ssp_getLaplacian(prior_tot, str_behaviour='fromSingle');
        Jp = ssp_assembleJ(J.asformat(spformat), L.multiply(alpha),L.multiply(alpha), L*0, L*0);
        '''
        elif global_behav == 'fromDiags':
        
            L = [None]*4;
            for i_prior in range(0,4):
                L[i] = ssp_getLaplacian(prior_tot[:,:,:,:, (6*i_prior): (6*(i_prior+1))], str_behaviour = 'fromDiags');
            Jp = ssp_assembleJ(J, L[0].multiply(alpha), L[1].multiply(alpha), L[2].multiply(alpha), L[3].multiply(alpha));
        #solve
        '''
        out[:] = np.float32(ssplin.lsqr(Jp, datap)[0]);
    return np.array(out)

def weightLap(x):
    px,py, pz = tf3Dgradient(0.1+10*x)
    p = tf.square(px) + tf.square(py) +tf.square(pz)
    kap = tf.sqrt(tf.exp(tf.divide(tf.negative(5 * p), tf.reduce_max(p, axis=[1, 2, 3]))))
    return kap

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
    return 1 - tf.divide(2*TC,TC+1)#1 - TC#tf.divide(2*TC,TC+1)


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    return tf.py_function(func, inp, Tout)

def tf3Dgradient(p):
    # x = [batch, length, width, depth, chan]
    p0 = tf.pad(p, [[0,0],[1,1],[1,1],[1,1],[0,0]]);
    gx = tf.manip.roll(p0, shift=1, axis = 1) - p0;
    gy = tf.manip.roll(p0, shift=1, axis = 2) - p0;
    gz = tf.manip.roll(p0, shift=1, axis = 3) - p0;
    return gx[:,1:-1,1:-1,1:-1,:],gy[:,1:-1,1:-1,1:-1,:], gz[:,1:-1,1:-1,1:-1,:]

def normalise_old(x):
    mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3]);
    outb = tf.div( tf.subtract(x, mean), 5*tf.sqrt(var))
    return tf.clip_by_value(outb,-1,1)

def normalise_old(x, type='a'):
    median = tf.contrib.distributions.percentile(x, 50.0);
    x = tf.subtract(x, median);
    if type=='a':
        prcmina=5*tf.nn.sigmoid(bias_variable([1]));
        prcmaxa=100 - 5*tf.nn.sigmoid(bias_variable([1]));
        div = tf.math.maximum(tf.abs(tf.contrib.distributions.percentile(x, prcmina)),tf.abs(tf.contrib.distributions.percentile(x, prcmaxa)));
    elif type=='s':
        prcmins=5*tf.nn.sigmoid(bias_variable([1]));
        prcmaxs=100 - 5*tf.nn.sigmoid(bias_variable([1]));
        div = tf.math.maximum(tf.abs(tf.contrib.distributions.percentile(x, prcmins)),tf.abs(tf.contrib.distributions.percentile(x, prcmaxs)));

    outb = tf.abs(tf.div(x, div))
    return tf.clip_by_value(outb, 0, 1)
def normalise(x, prcmin,prcmax):
    median = tf.contrib.distributions.percentile(x, 50.0);
    x = tf.subtract(x, median);
    div = tf.math.maximum(tf.abs(tf.contrib.distributions.percentile(x, prcmin)),tf.abs(tf.contrib.distributions.percentile(x, prcmax))); 
    outb = tf.abs(tf.div(x, div))
    #x = tf.abs(x)
    #x = tf.div(tf.subtract(x,tf.contrib.distributions.percentile(x, prcmin)),tf.subtract(tf.contrib.distributions.percentile(x, prcmax),tf.contrib.distributions.percentile(x, prcmin)) )
    return tf.clip_by_value(outb, 0, 1)

    #outb = tf.subtract(x, tf.reduce_mean(x, axis=None));
    #outb = tf.div(outb, 0.5*(tf.reduce_max(tf.abs(outb), axis=None)));
    #return outb
def to_bin(x):
    return tf.clip_by_value(tf.round(x), 0, 1)

def norm0to1(x, type='a'):
    # check if tensor is not flat and normalise
    if type=='a':
#        prcmin=5*tf.nn.sigmoid(bias_variable([1]))[0];
#        prcmax=100 - 5*tf.nn.sigmoid(bias_variable([1]))[0];
        prcmin= 1;
        prcmax= 99;
        
    elif type=='s':
        prcmin=2;
        prcmax=98;
        
    out = tf.cond(tf.greater(tf.reduce_max(x),tf.reduce_min(x)), lambda: normalise(x,prcmin, prcmax), lambda: x)
    return out

def NanTo0(x):
    out = tf.where(tf.is_nan(x), tf.zeros_like(x), x);
    out = tf.where(tf.is_inf(out), tf.zeros_like(x), out);
    return out;


def main(filePath, fileOutName, dataSetTrain, dataSetTest, tRand, MatOutName):
    iterMain = int(1)
    maxIter = int(80000)

    print('--------------------> DiffNet Init <--------------------')
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=tf.GPUOptions(allow_growth=True))
    #gpu_options=tf.GPUOptions(allow_growth=True)
    #config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    dataDbar = [None]*len(dataSetTest);
    dataDbar[0] = loadDiff.read_data_sets(dataSetTrain, dataSetTest[0], '');
    for i_datasets in range(1, len(dataSetTest)):
        dataDbar[i_datasets] = loadDiff.read_data_sets('absent', dataSetTest[i_datasets], '');

    muarecon0 = tf.placeholder(tf.float32, [bSize, Nx,Ny,Nz, chan])
    musrecon0 = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    truea = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    trues = tf.placeholder(tf.float32, [bSize, Nx, Ny, Nz, chan])
    #s = tf.placeholder(tf.float32, [bSize, nfreq,nfreq])
    #U = tf.placeholder(tf.float32, [bSize, Nt*Nm, nfreq])
    #V = tf.placeholder(tf.float32, [bSize,Nx*Ny*Nz*2,nfreq])
    #data = tf.placeholder(tf.float32, [bSize,Nt*Nm,1]);
    prior2D = tf.placeholder(tf.float32, [bSize, NxUS,NzUS])
    prior3D = tf.placeholder(tf.float32, [bSize, Nx,Ny,Nz,1]);
    coeff =  tf.placeholder(tf.float32, [4])


    #prior3D = tf.expand_dims(prior3D0, axis = 4)
    #prior2D = tf.placeholder(tf.float32, [bSize, Nx,Nz]);

    with tf.name_scope('Net'):
        #tf.tile( tf.eye(Nx*Nz, num_columns=Nz*Nx), [Ny,1] )
        if float(sys.argv[3]) == 0:
            prior2D = tf.zeros_like(prior2D)

        b0 = bias_variable([1])
        _,_,_,_,prior2Dconv = utun.UNETd(tf.expand_dims(prior2D, axis = 3))
        NxUSconv = int(prior2Dconv.get_shape()[1])
        NzUSconv = int(prior2Dconv.get_shape()[2])
        NcUSconv = int(prior2Dconv.get_shape()[3])

        #prior2Dconv = prior2D
        W2D3D = (1 )*tf.get_variable('W2D3D',shape=[Nx*Ny*Nz,NxUSconv*NzUSconv*NcUSconv], initializer=tf.contrib.layers.xavier_initializer())
        prior0 =(tf.matmul(tf.tile(tf.expand_dims( W2D3D, axis = 0),[bSize, 1,1]),
                                      tf.reshape(prior2Dconv, [bSize,-1,1]),
                                    )+b0)
        #recon_prior = tf.transpose(tf.reshape(prior0 , [bSize, Nx,Nz,Ny,1]), [0,1,3,2,4])
        recon_prior = tf.transpose(tf.reshape(prior0, [bSize,Nz, Ny, Nx, 1]), [0, 3, 2, 1, 4])
        recon_prior = NanTo0(recon_prior);
        Nmax = int(sys.argv[5])
        rel_loss_p = [None]*(Nmax+1)
        rel_loss_p[0] = coeff[0] * (tf.norm(tf.subtract(recon_prior, prior3D)) / (tf.norm(0.0+prior3D)))
        rel_loss_p[0] = 0.05*rel_loss_p[0] + coeff[0] * (0.9* overlap(prior3D, recon_prior) +
                                                         0.1* overlap(to_bin(prior3D), to_bin(recon_prior)));

        #Vt = tf.transpose(V[:, :, :], [0, 2, 1]);
        #J0 = tf.to_float(tf.matmul(U[:, :, :], tf.matmul(s[:, :, :], Vt)));
        #alpha = tf.sqrt(0.05* s[:, 0, 0]);

        inmuarecon0 = tf.identity(muarecon0)
        inmusrecon0 = tf.identity(musrecon0)

        if float(sys.argv[2])  != 0: # check if we want to consider the intitial reconstruction
            inmuarecon0 = NanTo0(norm0to1(inmuarecon0,type='a'))
            inmusrecon0 = NanTo0(norm0to1(inmusrecon0,type='s'))
        else:
            inmuarecon0 = tf.zeros_like(inmuarecon0)
            inmusrecon0 = tf.zeros_like(inmusrecon0)


        for i_rec in range(0,Nmax):
            bb = bias_variable([1])
            recon_prior = tf.nn.elu(UNET(inmuarecon0, inmusrecon0, tf.tanh(recon_prior))+ bb);
            recon_prior = NanTo0(recon_prior);
            # compute loss
            rel_loss_p[i_rec + 1] = coeff[i_rec + 1] * (tf.norm(tf.subtract(recon_prior, prior3D)) / (tf.norm(0.0+prior3D)))
            rel_loss_p[i_rec + 1] = 0.05*rel_loss_p[i_rec + 1] + coeff[i_rec + 1] * (0.9* overlap(prior3D, recon_prior) +
                                                                                     0.1* overlap(to_bin(prior3D), to_bin(recon_prior)));
            # compute optical reconstruction
            # check if loss depends on this calculation
            out_mua = tf.zeros_like(inmuarecon0);
            out_mus = tf.zeros_like(inmusrecon0);
            if Nmax > 1 :
                out = tf.cond(tf.equal(coeff[i_rec+1],0),
                              lambda: tf.zeros([bSize,Nx*Ny*Nz*2]),
                              lambda: tf.py_func(func = ssp_solve, inp = [J0,data,weightLap(tf.nn.relu(tf.transpose(recon_prior, [0,3,2,1,4]))), alpha, coeff[i_rec+1]], Tout= tf.float32))

                out_mua = tf.transpose(tf.reshape(out[:, 0:Nx * Ny*Nz], [bSize, Nz, Ny, Nx, 1]), [0, 3, 2, 1,4]);
                out_mus = tf.transpose(tf.reshape(out[:, Nx * Ny * Nz:], [bSize, Nz, Ny, Nx, 1]), [0, 3, 2, 1,4]);
                out_mua = 0.01 * (1 + out_mua);
                out_mus = tf.div(1.0, 3.0 * (1 + out_mus) / 3.0);
                inmuarecon0 = NanTo0(norm0to1(NanTo0(out_mua))); #after the first reconstruction we can input the
                inmusrecon0 = NanTo0(norm0to1(NanTo0(out_mus)));

            ''''
            control = weightLap(prior3D)
            outc = tf.py_func(func=ssp_solve, inp=[J0, data, weightLap(tf.nn.relu(tf.transpose(prior3D, [0,3,2,1,4]))), alpha], Tout=tf.float32)
            out_muac = tf.transpose(tf.reshape(outc[:, 0:Nx * Ny * Nz], [bSize, Nz, Ny, Nx, 1]), [0, 3, 2, 1,4]);
            out_musc = tf.transpose(tf.reshape(outc[:, Nx * Ny * Nz:], [bSize, Nz, Ny, Nx, 1]), [0, 3, 2, 1,4]);
            out_muac = 0.001 * (1 + out_muac);
            out_musc = tf.div(1.0, 3.0 * (1 + out_musc) / 3.0);
            '''


    #out_mus = out_mua;

    with tf.name_scope('optimizer'):
        #rel_loss_p = tf.norm(tf.subtract((recon_prior), prior3D)) / tf.norm(prior3D)
        rel_loss_a = tf.norm(tf.subtract(tf.nn.relu(out_mua), truea)) / tf.norm(truea)
        rel_loss_s = tf.norm(tf.subtract(tf.nn.relu(out_mus), trues)) / tf.norm(trues)
        rel_loss = tf.add_n(rel_loss_p) #+ rel_loss_p[1] + rel_loss_p[2] #rel_loss_a + 0.1*rel_loss_s#+0.4*rel_loss_p;


    with tf.name_scope('summaries'):
        tf.summary.scalar('rel_loss_p', rel_loss_p)
        tf.summary.scalar('rel_loss_s', rel_loss_s)
        tf.summary.scalar('rel_loss_a', rel_loss_a)
        tf.summary.scalar('rel_loss', rel_loss)

        expName = [None]*len(fileOutName);

        for i_expName in range(0, len(fileOutName)):
            expName[i_expName] = 'DiffNet_' + fileOutName[i_expName];

        with tf.name_scope('training'):
            learningRate = tf.constant(1e-6)

            train_step = tf.train.AdamOptimizer(learningRate).minimize(rel_loss)

    sess.run(tf.global_variables_initializer())
    lVal = 2e-4
    ''''
    feed_test = {recona: dataDbar[0].test.recona[0:bSize],
                 recons: dataDbar[0].test.recons[0:bSize],
                 prior3D0: dataDbar[0].test.prior[0:bSize],
                 prior2D: dataDbar[0].test.priorslice[0:bSize]}
    '''
    startIt = 0
    Ltrain = [0];
    Ltest = [0];
    if Nmax > 1:
        fact_maxiter = 4;
    else:
        fact_maxiter = 1;

    saver = tf.train.Saver()
    # initialize champion
    champion_loss_val = float(1);
    for i in range(maxIter):

        batch = dataDbar[0].train.next_batch(bSize)
        #      test1 = batch[0]
        #      test2 = batch[1]
        parint = randint(0,1);
        if  parint==0:

            feed_train = {muarecon0: float(sys.argv[2]) * np.float32(np.array(batch['recona'])),
                          musrecon0: float(sys.argv[2]) * np.float32(np.array(batch['recons'])),
                          prior3D: np.float32(np.array(batch['prior'])),
                          prior2D: float(sys.argv[3])*np.float32(np.array(batch['priorslice'])),
                          trues:np.float32(np.array(batch['trues'])),
                          truea:np.float32(np.array(batch['truea'])),
                          learningRate: lVal}
        elif parint==1:

            feed_train = {muarecon0: float(sys.argv[2]) * np.flip(np.float32(np.array(batch['recona'])),axis = 2) ,
                          musrecon0: float(sys.argv[2]) * np.flip(np.float32(np.array(batch['recons'])),axis = 2),
                          prior3D: np.flip(np.float32(np.array(batch['prior'])), axis = 2),
                          prior2D: float(sys.argv[3]) * np.float32(np.array(batch['priorslice'])),
                          trues: np.flip(np.float32(np.array(batch['trues'])),axis = 2),
                          truea: np.flip(np.float32(np.array(batch['truea'])),axis = 2),
                          learningRate: lVal}
        elif parint == 2:

            feed_train = {muarecon0: float(sys.argv[2]) * np.flip(np.float32(np.array(batch['recona'])), axis=1),
                          musrecon0: float(sys.argv[2]) * np.flip(np.float32(np.array(batch['recons'])), axis=1),
                          prior3D: np.flip(np.float32(np.array(batch['prior'])), axis=1),
                          prior2D: float(sys.argv[3]) * np.flip(np.float32(np.array(batch['priorslice'])), axis = 1),
                          trues: np.flip(np.float32(np.array(batch['trues'])), axis=1),
                          truea: np.flip(np.float32(np.array(batch['truea'])), axis=1),
                          learningRate: lVal}
        elif parint==3:

            feed_train = {muarecon0: float(sys.argv[2]) * np.flip(np.flip(np.float32(np.array(batch['recona'])),axis = 2) ,axis = 1),
                          musrecon0: float(sys.argv[2]) * np.flip(np.flip(np.float32(np.array(batch['recons'])),axis=2), axis = 1),
                          prior3D: np.flip(np.flip(np.float32(np.array(batch['prior'])),axis=2),axis=1),
                          prior2D: float(sys.argv[3]) * np.flip(np.float32(np.array(batch['priorslice'])),axis=1),
                          trues: np.flip(np.flip(np.float32(np.array(batch['trues'])),axis=2),axis=1),
                          truea: np.flip(np.flip(np.float32(np.array(batch['truea'])),axis=2),axis=1),
                          learningRate: lVal}

        if i <=200:
            feed_train[coeff] = np.array([1, 0, 0, 0]);
            _, loss_result_train = sess.run([train_step, rel_loss_p], feed_dict=feed_train);
        elif (i < maxIter/(fact_maxiter)):
            #feed_train[coeff] = np.array([0.01,1,0,0]);
            feed_train[coeff] = np.array([0.0001, 1, 0, 0]);
            _, loss_result_train = sess.run([train_step, rel_loss_p], feed_dict=feed_train);
        elif ((i >= maxIter/(fact_maxiter)) & (i < ((2*maxIter)/fact_maxiter))):
            feed_train[coeff] = np.array([0,0.5,0.5,0]);
            _, loss_result_train = sess.run([train_step, rel_loss_p], feed_dict=feed_train);
        elif ((i >=((2*maxIter)/fact_maxiter)) & (i < ((3*maxIter)/(fact_maxiter)))) :
            feed_train[coeff] = np.array([0, 0.333,0.333,0.333]);
            _, loss_result_train = sess.run([train_step, rel_loss_p], feed_dict=feed_train);
        elif (i >=((3*maxIter)/fact_maxiter)) :
            feed_train[coeff] = np.array([0, 0.1,0.1,1]);
            _, loss_result_train = sess.run([train_step, rel_loss_p], feed_dict=feed_train);



        if ((Nmax == 1) & ((i % 50) == 0)):
            print('proceeding 1 iter.. %s ---fact %s' % (str(i),str(fact_maxiter)))

        if ((Nmax == 3) & ((i % 10) == 0)):
            print('proceeding  3 iter.. %s ---fact %s' % (str(i),str(fact_maxiter)))

        if i == 20000:
            lVal = lVal/2
        if i == 40000:
            lVal = lVal/2
        if i == 50000:
            lVal = lVal/2;
        if i == 65000:
            lVal = lVal/2;

        #loss_result_train = multiple_run(feed_train, train_step, [muarecon0, musrecon0],[rel_loss_p], Niter = 2, Train = True)


        Ltrain.append(loss_result_train)

        if (((i % 25 == 0) & (i >= 12000))|((i % 100 == 0) & (i < 12000))) : #write test
            # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            for i_test in range(0,1):#len(dataDbar) - 1):
                tBeg = randint(0, tRand[i_test])
                tEnd = tBeg + bSize
                it = i + startIt

                feed_test = {muarecon0: float(sys.argv[2]) * np.float32(dataDbar[i_test].test.recona[tBeg:tEnd]),
                             musrecon0: float(sys.argv[2]) * np.float32(dataDbar[i_test].test.recons[tBeg:tEnd]),
                             prior3D: dataDbar[i_test].test.prior[tBeg:tEnd],
                             prior2D: float(sys.argv[3])* np.float32(dataDbar[i_test].test.priorslice[tBeg:tEnd]),
                             trues: dataDbar[i_test].test.trues[tBeg:tEnd],
                             truea: dataDbar[i_test].test.truea[tBeg:tEnd]}
                feed_test[coeff] = feed_train[coeff];
                #test_accuracy, test_result = sess.run([loss, prior_mua],
                #                                      feed_dict=feed_test)

                rel_loss_res_p,rel_loss_res_a,rel_loss_res_s= sess.run([ rel_loss_p,rel_loss_a,rel_loss_s],
                                                                            feed_dict=feed_test)

                Ltest.append(rel_loss_res_p)

                print('iter={}, rel.lossp={}, rel.lossa.test={}, rel.losss.test={}, rel.loss.tottrain={}'.format(i,rel_loss_res_p,rel_loss_res_a,rel_loss_res_s, loss_result_train))

            # run validation loss
            if i > 8000:
                loss_val = [None] * np.shape(dataDbar[1].test.recona)[0];
                for i_val in range(0, np.shape(dataDbar[1].test.recona)[0] - bSize+1):
                    #if i_val % 100 == 0:
                     #   print('Evaluating validation={}'.format(i_val))
                    tBeg = i_val;
                    tEnd = i_val + bSize;
                    feed_test = {muarecon0: float(sys.argv[2]) * np.float32(dataDbar[1].test.recona[tBeg:tEnd]),
                                 musrecon0: float(sys.argv[2]) * np.float32(dataDbar[1].test.recons[tBeg:tEnd]),
                                 prior3D: dataDbar[1].test.prior[tBeg:tEnd],
                                 prior2D: float(sys.argv[3]) * np.float32(dataDbar[1].test.priorslice[tBeg:tEnd]),
                                 trues: dataDbar[1].test.trues[tBeg:tEnd],
                                 truea: dataDbar[1].test.truea[tBeg:tEnd]  };
                    feed_test[coeff] = feed_train[coeff];
                    validation_loss = sess.run([rel_loss], feed_dict=feed_test)
                    loss_val[i_val] = np.sum(validation_loss[-1]);
                    if np.isfinite(loss_val[i_val])==False:
                        loss_val[i_val] = np.NaN;
                mean_val_loss = np.nanmean(np.array(loss_val));
                print('Mean Validation is = {}'.format(mean_val_loss))
                 # if validation has lower error, save output dataset
                if (mean_val_loss < champion_loss_val):
                    champion_loss_val = np.nanmean(np.array(loss_val));
                    print('NEW CHAMPION, SAVING ...')
                    saved_path = saver.save(sess, MatOutName[0][0:-4] + 'best_binaryInput')
                    saver.restore(sess, saved_path);
    for i_dataset in range(0, len(dataDbar)):
        resulta = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])
        results = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])
        resultp = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])
        STresult = [None] * (np.shape(dataDbar[i_dataset].test.recona)[0])
        print('saving test = {}'.format(i_dataset))
        for i_final in range(0, np.shape(dataDbar[i_dataset].test.recona)[0] - bSize+1 ):
            #if i_final % 200 == 0:
             #   print('Evaluating test={}'.format(i_final))
            tBeg = i_final;
            tEnd = i_final + bSize;
            feed_test = {muarecon0:  float(sys.argv[2]) * np.float32(dataDbar[i_dataset].test.recona[tBeg:tEnd]),
                         musrecon0: float(sys.argv[2])*np.float32(dataDbar[i_dataset].test.recons[tBeg:tEnd]),
                         prior3D: dataDbar[i_dataset].test.prior[tBeg:tEnd],
                         prior2D: float(sys.argv[3])* np.float32(dataDbar[i_dataset].test.priorslice[tBeg:tEnd]),
                         trues: dataDbar[i_dataset].test.trues[tBeg:tEnd],
                         truea: dataDbar[i_dataset].test.truea[tBeg:tEnd]
                         };

            feed_test[coeff] = feed_train[coeff];
            test_resultp,test_STresult, test_resulta,test_results = sess.run([recon_prior,prior0,out_mua,out_mus], feed_dict=feed_test);
            #test_resulta, test_results = multiple_run([out_mua, out_mus], feed_dict=feed_test)

            resulta[i_final:i_final + bSize] = test_resulta[0:];
            results[i_final:i_final + bSize] = test_results[0:];
            resultp[i_final:i_final + bSize] = test_resultp[0:];
            STresult[i_final:i_final + bSize] = test_STresult[0:];
        dict_sio = {'STresult': STresult,
                    'resultp': resultp,
                    'trainLoss':Ltrain,
                    'test_Loss':Ltest}
        # print(MatOutName[i_dataset])
        # ;
        print('Saving Matlab file')
        sio.savemat(MatOutName[i_dataset], dict_sio)
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
specName_train = [ str(sys.argv[4])+str(sys.argv[6])];
#specName_test =  [[str(sys.argv[4]),str(sys.argv[4])+'_v', str(sys.argv[4])+'_EXPCIL']];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];

specName_test = [[str(sys.argv[4])+str(sys.argv[6]),
                  str(sys.argv[4])+str(sys.argv[6]) + '_v',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL_kwave',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL_simBORN',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL_kwave_simBORN',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL_simFEM',
                  'BORNsvdLarge2'+str(sys.argv[6]) + '_EXPCIL_kwave_simFEM',
                  'BORNsvdLarge2'+str(sys.argv[6])+ '_kwave_sgm',
                  'FEMsvdLarge2'+str(sys.argv[6])+ '_kwave_sgm',
                  'FEMHETEsvdLarge2'+str(sys.argv[6]) + '_kwave_sgm',
                  'BORNsvdLarge2'+str(sys.argv[6]) ,
                  'FEMsvdLarge2'+str(sys.argv[6]),
                  'FEMHETEsvdLarge2'+str(sys.argv[6]),
                  'FEMcyl'+str(sys.argv[6]) ]];

#specName_train = [ 'allRandom' ,'nohete', 'from1to4' ];
#specName_test = [['allRandom','from1to4','nohete'] ,['nohete', 'from1to4', 'allRandom'],['from1to4','nohete', 'allRandom']];#,['2_t','1_t', '_merged_t'], ['_merged_t','2_t','1_t']];
netPath = 'netData/test_DOCM.ckpt';
train_num = [ np.multiply( [ 200 - bSize -1, 200 - bSize -1,72-bSize -1,200,200],1)];
saveMatCommon = 'RESULTS/'+'DOCM_orig_sepTest/';#implicit'
logNameCommon = folder_tensorboard +'3UNET__';

print('Running main code')
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

python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_meanvar_50k_dice' 1 1 'BORNsvdLarge2' 1 ; 
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic_meanvar_50k_dice' 1 0 'BORNsvdLarge2' 1 ; 
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice_meanvar_50k_dice' 0 1 'BORNsvdLarge2' 1 ;
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw2' 1 1 'BORNsvdLarge2USonly' 1 ; 
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice_withEXP_frmw2' 0 1 'BORNsvdLarge2USonly' 1 ;

python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw3' 1 1 'BORNsvdLarge2USonly' 1 ; 
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice_withEXP_frmw3' 0 1 'BORNsvdLarge2USonly' 1 ;

'''''
