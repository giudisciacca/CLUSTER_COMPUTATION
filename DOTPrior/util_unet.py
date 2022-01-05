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
import scipy.io as sio  # %ok- 1
import random
random.seed(10)
from random import randint

from scipy.sparse import linalg as ssplin
import scipy as sc
# from numba import cuda
import matplotlib.pyplot as plt
import multiprocessing

bSize = int(1)
chan = int(1)
wchan = int(4);

Nx = int(32);
Ny = int(1)
Nz = int(16)
Nt = int(10);
nsource = int(8);
ndetector = int(8);
nfreq = 20;
Nm = int(nsource * (ndetector-1))
size_domain = Nx*Nz;

pad = 'VALID';

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
    #x = tf.truncated_normal(shape, stddev=0.005)
    x =  np.sqrt(2/(shape[0]*shape[1]*(shape[2]+shape[3])));
    #initializer = tf.random_uniform(shape, minval=-x, maxval=x);
    initializer = tf.truncated_normal(shape, stddev=x);
    return tf.get_variable(layernum, initializer=initializer)

def weight_variable2(shape, layernum):
    #initializer = tf.random_uniform(shape, minval=-x, maxval=x);
    initializer = tf.truncated_normal(shape, stddev= 0.1);
    return tf.get_variable(layernum,initializer=initializer)


def bias_variable(shape):
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padconv ='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, filter = W, padding=padconv, strides=[1, 1, 1, 1] )

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
    W[0] = weight_variable([3, 3, in_channels, out_channels], str(laynum) + '1'+str(randint(0, 100000)))
    W[1] = weight_variable([3, 3, out_channels, out_channels], str(laynum) + '2'+str(randint(0, 100000)))
    W[2] = weight_variable([3, 3, out_channels, out_channels], str(laynum)+'3'+str(randint(0, 100000)))
    return W

def u_net_weight_up(laynum, in_channels, out_channels):

    W = [None]*3;
    W[0] = weight_variable([3, 3, out_channels, in_channels], str(laynum) + '1'+str(randint(0, 100000)))
    #W[0] = weight_variable([3, 3, in_channels ,out_channels], str(laynum) + '1'+str(randint(0, 100000)))
    W[1] = weight_variable([3, 3, out_channels, out_channels], str(laynum) + '2'+str(randint(0, 100000)))
    W[2] = weight_variable([3, 3, out_channels, out_channels], str(laynum) + '3'+str(randint(0, 100000)))
    return W

def u_net_step_down(x, W):

    b = [None] * 3;
    b[0] = bias_variable([W[0].get_shape()[3]])
    b[1] = bias_variable([W[1].get_shape()[3]])
    b[2] = bias_variable([W[2].get_shape()[3]])
    l1 = tf.nn.elu(conv2d(x,W[0])) + b[0];
    l2 = tf.nn.elu(conv2d(l1,W[1])) + b[1];
    l3 = conv2d(l2, W[2]);
    out = tf.nn.elu(tf.add(l3, l1));
    #out = tf.nn.max_pool( tf.nn.elu(tf.add(l3, l1)), [1,4,4,1], [1,2,2,1], padding='VALID') ;
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    out = max_pool_2d(out) + b[2];
    return out;


def u_net_step_up(x, W, output_shape):
    b = [None] * 3;
    b[0] = bias_variable([W[0].get_shape()[2]])
    #b[0] = bias_variable([W[0].get_shape()[3]])
    b[1] = bias_variable([W[1].get_shape()[3]])
    b[2] = bias_variable([W[2].get_shape()[3]])
    l1 = tf.nn.relu(d_conv2d(x, W[0], output_shape)) + b[0];
    #l1 = tf.nn.elu(conv2d(x, W[0])) + b[0];
    l2 = tf.nn.elu(conv2d(l1, W[1]))+ b[1];
    l3 = conv2d(l2, W[2]);
    #out = tf.nn.elu(tf.add(l3, l1))+ b[2];
    out = (tf.add(l3 +b[2], l1));
    #out = tf.nn.max_pool3d(tf.nn.elu(tf.add(l3, l1)), [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME') +b[2]
    return out

def UNETd(inValue, chan_net = chan):
    W0 = weight_variable([3,3,chan_net,int(wchan/4)], 'UNETd'+str(randint(0, 100000)));
    l0 = conv2d(inValue, W0);
    W1_d = u_net_weight_down(randint(0, 100000), int(int(int(wchan)/4)), int(int(wchan)/4))
    b1_d = bias_variable([int(int(wchan)/4)]);
    l1_d = u_net_step_down(l0, W1_d) + b1_d #32
    W2_d = u_net_weight_down(randint(0, 100000), int(int(wchan)/4), int(int(wchan)/2))
    b2_d = bias_variable([int(int(wchan)/2)]);
    l2_d = u_net_step_down(l1_d, W2_d) + b2_d#64
    W3_d = u_net_weight_down(randint(0, 100000), int(int(wchan)/2), int(wchan))
    b3_d = bias_variable([int(wchan)]);
    l3_d = u_net_step_down(l2_d, W3_d) + b3_d #128
    W4_d = u_net_weight_down( randint(0, 100000), int(wchan), int(1) )#'int(wchan))
    b4_d = bias_variable([int(wchan)]);
    l4_d = u_net_step_down(l3_d, W4_d) + b4_d #128
    return tf.nn.elu(l0),tf.nn.elu(l1_d), tf.nn.elu(l2_d), tf.nn.elu(l3_d),tf.nn.elu(l4_d)

def UNET(inValue0, inValue1, inValue2, outchan=1, outShape=None):

    if outShape == None:
        outShape = tf.get_shape(inValue0);

    l0 = 0;

    _,a0, b0, c0, d0 = UNETd(inValue0)
    _,a1, b1, c1, d1 = UNETd(inValue1)
    _,a2, b2, c2, d2 = UNETd(inValue2)

    wout4 = weight_variable([3, 3, 3*int(wchan), int(wchan)], 'zw101010'+str(randint(0, 100000)));
    l4_d = conv2d(tf.concat([ d0, d1, d2], axis = 3), wout4);
    wout3 = weight_variable([3,3,3*int(wchan),int(wchan)], 'zw10101'+str(randint(0, 100000)));
    l3_d = conv2d(tf.concat([ c0, c1, c2], axis = 3), wout3);
    wout2 = weight_variable([3,3,3*int(int(wchan)/2),int(int(wchan)/2)], 'zw10102'+str(randint(0, 100000)));
    l2_d = conv2d(tf.concat([ b0, b1,b2], axis = 3), wout2);
    wout1 = weight_variable([3,3,3*int(int(wchan)/4),int(int(wchan)/4)], 'zw10103'+str(randint(0, 100000)));
    l1_d = conv2d(tf.concat([ a0, a1,a2], axis = 3), wout1);#a2#+a1+a2#tf.concat([ a0, a1, a2], axis = 4) ;

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
    l1_u = tf.add(u_net_step_up(l4_d, W1_u, [bSize, Nx, Nz, int(wchan)]), l4_d) + b1_u ;#128
    W2_u = u_net_weight_up(6, int(wchan), int(wchan))
    b2_u = bias_variable([int(wchan)]);
    l2_u = tf.add(u_net_step_up(l1_u, W2_u, [bSize, Nx, Nz, int(wchan)]),l3_d) + b2_u ; #128
    W3_u = u_net_weight_up(7, int(wchan), int(int(wchan)/2))
    b3_u = bias_variable([int(int(wchan)/2)])
    l3_u = tf.add(u_net_step_up(l2_u, W3_u, [bSize, Nx,  Nz, int(int(wchan)/2)]), l2_d) + b3_u ;#64
    W4_u = u_net_weight_up(8, int(int(wchan)/2), int(int(wchan)/4))
    b4_u = bias_variable([int(int(wchan)/4)]);
    l4_u = tf.add(u_net_step_up(l3_u, W4_u, [bSize, Nx, Nz, int(int(wchan)/4)]),l1_d) + b4_u ;
    Wout_u = u_net_weight_up(9, int(int(wchan)/4), int(int(wchan)/4))
    bout_u = bias_variable([int(int(wchan)/4)]);
    out = tf.add(u_net_step_up(l4_u, Wout_u, [bSize, Nx, Nz, int(int(wchan)/4)]), l0) + bout_u;
    wout = weight_variable([3,3,int(int(wchan)/4),outchan], '101'+str(randint(0, 100000)));
    out = conv2d(out, wout);
    wout_o = weight_variable([7, 7, outchan+3*chan, outchan], 'o10100011'+str(randint(0, 100000)));
    return d_conv2d( tf.concat([out, inValue0, inValue1, inValue2], axis = 3), wout_o, output_shape = outShape)
#    wout_o = weight_variable([3, 3, 3, 2, 1], 'o10100011');
#    return conv3d( tf.concat([out, inValue2], axis = 4), wout_o)