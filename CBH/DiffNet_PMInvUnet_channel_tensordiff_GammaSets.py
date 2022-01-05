#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:52:51 2019

@author: gdisciac
"""

## DiffNet Script with only one CNN to get Gamma
## 9-POINT STENCIL

import tensorflow as tf
import sys
sys.path.append("..")
import Load_DiffNet_Inv as loadDiff
import numpy as np
import utilTemp_nonlinInv as util
import os
#import scipy
from random import randint
import scipy.io as sio
import multiprocessing

#import matplotlib.pyplot as plt

FLAGS = None

name = os.path.splitext(os.path.basename(__file__))[0]

bSize=int(16)
chan=int(9)

N=int(60)
NN=N*N
    
zero=tf.constant(0.0, shape=[bSize,1])
zeroNHorz=tf.constant(0.0, shape=[bSize,N,1,1])
zeroNVert=tf.constant(0.0, shape=[bSize,1,N,1])
    

def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


#dropOut=0.01
def kappaEstimator(x_in):  
    x_in = tf.reshape(x_in,[bSize,N,N,1])
    kappaEst=tf.contrib.layers.conv2d(x_in,32,3)
    kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
    kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
             
    kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)        

    return kappaEst
       
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.025, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  """conv3d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def d_conv2d(x, W, output_shape):
  """conv3d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d_transpose(x, W,output_shape, strides=[1, 1, 1, 1], padding='SAME')

def u_net_step_down(x, w1, w2, w3):
    l1 = tf.nn.relu( conv2d(x, w1));
    l2 = tf.nn.relu(conv2d(l1, w2));
    l3 = conv2d(l2, w3);
    out = tf.nn.relu(tf.add(l3,l1));
    return out;

def u_net_step_up(x, w1, w2, w3, output_shape):
    l1 = tf.nn.relu( d_conv2d(x, w1, output_shape));
    l2 = tf.nn.relu(conv2d(l1, w2));
    l3 = conv2d(l2, w3);
    out = tf.nn.relu(tf.add(l3,l1));
    return out;

    
def diffLayer(x_in,bSize,N,layNum):
#    NN=N*N    

    x_full=tf.reshape(x_in ,[bSize,N,N,chan])
    Kappa = [];
    x_update=x_full[:,:,:,0]
    x_update=tf.reshape(x_update,[bSize,N,N,1])
    dt = tf.constant(0.2, shape=[1], dtype = tf.float32)
    dt = tf.Variable(dt,name='dt_' + str(layNum));           
    #First network initialises variables    
    #kappa=kappaEstimator(x_update)
    with tf.name_scope('NetworkInit'):
#        kappaEst=tf.contrib.layers.conv2d(x_in,32,3)        
        W_convL1 = weight_variable([3, 3, 1, 32])
        b_convL1 = bias_variable([32])
        kappaEst = tf.nn.relu(conv2d(x_update, W_convL1) + b_convL1)
        
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL2 = weight_variable([3, 3, 32, 64])
        b_convL2 = bias_variable([64])
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL2) + b_convL2)
        
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        W_convL3 = weight_variable([3, 3, 64, 64])
        b_convL3 = bias_variable([64])
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL3) + b_convL3)
        
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)
        
        W_convL4 = weight_variable([3, 3, 64, 9])
        b_convL4 = bias_variable([9])
        kappa = (conv2d(kappaEst, W_convL4) + b_convL4)
        Kappa.append(kappa);
            
    kapDiag      = tf.reshape(kappa[:,:,:,0],[bSize,N,N,1])
    kapUp        = tf.reshape(kappa[:,:,:,1],[bSize,N,N,1])
    kapDown      = tf.reshape(kappa[:,:,:,2],[bSize,N,N,1])
    kapLeft      = tf.reshape(kappa[:,:,:,3],[bSize,N,N,1])
    kapRight     = tf.reshape(kappa[:,:,:,4],[bSize,N,N,1])
    kapUpRight   = tf.reshape(kappa[:,:,:,5],[bSize,N,N,1])
    kapUpLeft    = tf.reshape(kappa[:,:,:,6],[bSize,N,N,1])
    kapDownRight = tf.reshape(kappa[:,:,:,7],[bSize,N,N,1])
    kapDownLeft  = tf.reshape(kappa[:,:,:,8],[bSize,N,N,1])

    xUp = tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)
    xDown = tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)
    xLeft = tf.concat([zeroNHorz,x_update[:,:,0:N-1]],axis=2)
    xRight = tf.concat([x_update[:,:,1:N],zeroNHorz],axis=2)
    xUpRight = tf.concat([tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)[:,:,1:N],zeroNHorz],axis=2)
    xDownLeft = tf.concat([zeroNHorz,tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)[:,:,0:N-1]],axis=2)
    xUpLeft = tf.concat([zeroNHorz,tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)[:,:,0:N-1]],axis = 2)
    xDownRight = tf.concat([tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)[:,:,1:N],zeroNHorz],axis=2)
    
    xDiag  = tf.multiply(kapDiag,x_update)
    xUp    = tf.multiply(kapUp,xUp)
    xDown  = tf.multiply(kapDown,xDown)
    xLeft  = tf.multiply(kapLeft,xLeft)
    xRight = tf.multiply(kapRight,xRight)
    xUpRight = tf.multiply(kapUpRight,xUpRight)
    xDownLeft = tf.multiply(kapDownLeft,xDownLeft)
    xUpLeft = tf.multiply(kapUpLeft,xUpLeft);
    xDownRight = tf.multiply(kapDownRight,xDownRight);
    
    x_update = x_update + dt*(xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight) #Test +xDiag
    x_out = x_update
            
    for xi in range(chan-1):
        x_update=x_full[:,:,:,xi+1]
        x_update=tf.reshape(x_update,[bSize,N,N,1]) 
#        kappa=kappaEstimator(x_update)
        ''' Network begin: variables already existing '''      
#        kappaEst=tf.contrib.layers.conv2d(x_in,32,3)        
        kappaEst = tf.nn.relu(conv2d(x_update, W_convL1) + b_convL1)
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL2) + b_convL2)        
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,32,3)
        kappaEst = tf.nn.relu(conv2d(kappaEst, W_convL3) + b_convL3)
#        kappaEst=tf.contrib.layers.conv2d(kappaEst,5,3,activation_fn=None)
        kappa = (conv2d(kappaEst, W_convL4) + b_convL4)
        Kappa.append(kappa);
        ''' Network end '''
        
        kapDiag  = tf.reshape(kappa[:,:,:,0],[bSize,N,N,1])
        kapUp    = tf.reshape(kappa[:,:,:,1],[bSize,N,N,1])
        kapDown  = tf.reshape(kappa[:,:,:,2],[bSize,N,N,1])
        kapLeft  = tf.reshape(kappa[:,:,:,3],[bSize,N,N,1])
        kapRight = tf.reshape(kappa[:,:,:,4],[bSize,N,N,1])
        kapUpRight = tf.reshape(kappa[:,:,:,5],[bSize,N,N,1])
        kapUpLeft = tf.reshape(kappa[:,:,:,6],[bSize,N,N,1])
        kapDownRight = tf.reshape(kappa[:,:,:,7],[bSize,N,N,1])
        kapDownLeft = tf.reshape(kappa[:,:,:,8],[bSize,N,N,1])
    
        xUp = tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)
        xDown = tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)
        xLeft = tf.concat([zeroNHorz,x_update[:,:,0:N-1]],axis=2)
        xRight = tf.concat([x_update[:,:,1:N],zeroNHorz],axis=2)
        xUpRight = tf.concat([tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)[:,:,1:N],zeroNHorz],axis=2)
        xDownLeft = tf.concat([zeroNHorz,tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)[:,:,0:N-1]],axis=2)
        xUpLeft = tf.concat([zeroNHorz,tf.concat([zeroNVert,x_update[:,0:N-1,:]],axis=1)[:,:,0:N-1]],axis = 2)
        xDownRight = tf.concat([tf.concat([x_update[:,1:N,:],zeroNVert],axis=1)[:,:,1:N],zeroNHorz],axis=2)
            
        xDiag  = tf.multiply(kapDiag,x_update)
        xUp    = tf.multiply(kapUp,xUp)
        xDown  = tf.multiply(kapDown,xDown)
        xLeft  = tf.multiply(kapLeft,xLeft)
        xRight = tf.multiply(kapRight,xRight)
        xUpRight = tf.multiply(kapUpRight,xUpRight)
        xDownLeft = tf.multiply(kapDownLeft,xDownLeft)
        xUpLeft = tf.multiply(kapUpLeft,xUpLeft);
        xDownRight = tf.multiply(kapDownRight,xDownRight);
        
        x_update = x_update + dt*(xUp + xDown + xLeft + xRight - xDiag + xUpRight + xDownLeft - xUpLeft - xDownRight) #Test +xDiag

        x_out = tf.concat([x_out,x_update],axis=3)

        
#    diagFac = getKappa(4.0, [N*N],'diagFac_' + str(layNum))
#    xDiag=tf.reshape(xDiag,[bSize,N*N])    
#    xDiag  = tf.multiply(diagFac,xDiag)
#    xDiag=tf.reshape(xDiag,[bSize,N,N,1])    

    return x_out, kappa, Kappa


def diffLayer_channel(x_in,bSize,N,chan):
    
    x_chan=tf.expand_dims(diffLayer(x_in[:,:,:,0],bSize,N),2)
    
    for bbb in range(chan-1):
        x_chan=tf.concat([x_chan,tf.expand_dims(diffLayer(x_in[:,:,:,bbb+1],bSize,N),2)],axis=2)    
        
    x_chan = tf.reshape(x_chan,[bSize,N,N,chan])
    return x_chan
        
        
    

def getKappa(inVal,shape,varName):
    kappa = tf.constant(inVal, shape=shape)
    return tf.Variable(kappa, name=varName)



def main(filePath,fileOutName,dataSetTrain,dataSetTest, tRand,MatOutName):
          
    iterMain=int(1)    
#    dataSetTest  =  'data/diffNet_nonlin_4step_test.mat'
#    dataSetTrain  =  'data/diffNet_nonlin_4step_train.mat'    
#    fileOutName = '
#    filePath = 'netData/diffNet_NonLinear_Inv_PM_4step.ckpt'           
#    bSize=int(bSizeIn)
    maxIter=int(35000)           
    print('--------------------> DiffNet Init <--------------------')
    #config = tf.ConfigProto();
    #%config.gpu_options.allow_growth = True;       
    #sess = tf.InteractiveSession(config = config)    
    #sess = tf.Session();
    sess = tf.InteractiveSession()     
    dataDbar = loadDiff.read_data_sets(dataSetTrain,dataSetTest)
    imSize=dataDbar.train.true.shape
      
    # Create the model
    imag = tf.placeholder(tf.float32, [None, imSize[1],imSize[2],chan])
    true = tf.placeholder(tf.float32, [None, imSize[1],imSize[2]])    
        
    with tf.name_scope('DiffNet'):      
      x_update=tf.reshape(imag ,[bSize,N,N,chan])      
      iii = 0;
      for iii in range(iterMain):
          x_update, kappaEst1, Kappa1 = diffLayer(x_update,bSize,N,0 + 10 * iii)
          x_update, kappaEst2, Kappa2 = diffLayer(x_update,bSize,N,1 + 10 * iii)
          x_update, kappaEst3, Kappa3 = diffLayer(x_update,bSize,N,2 + 10 * iii)  
          x_update, kappaEst4, Kappa4 = diffLayer(x_update,bSize,N,3 + 10 * iii)  
          x_update, kappaEst5, Kappa5 = diffLayer(x_update,bSize,N,4 + 10 * iii)  

      
      x_update=tf.reshape(x_update,[bSize,N,N,chan])  
      x_sum = x_update[:,:,:,0]
      for ccc in range(chan-1):
          x_sum = x_sum + x_update[:,:,:,ccc+1]
      x_sum=x_sum/chan
      y_diff = tf.nn.relu(x_sum)    
          
    saver = tf.train.Saver()
        
    with tf.name_scope('optimizer'):         
         loss = tf.norm(tf.subtract(tf.nn.relu(true),y_diff))/float(bSize)
         rel_loss = tf.norm(tf.subtract(tf.nn.relu(true),y_diff))/tf.norm(tf.nn.relu(true))
    #     added_loss = -tf.scalar_mul(100.0,tf.minimum( tf.subtract(tf.norm(y_diff),10.0),0.0)) 
    #    global_step = tf.Variable(0, trainable=False)   
         learningRate=tf.constant(1e-3)
         train_step = tf.train.AdamOptimizer(learningRate).minimize(loss)
        
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('rel_loss', rel_loss)
        tf.summary.scalar('psnr', psnr(y_diff, true))
    
        tf.summary.image('result', tf.reshape(y_diff[1],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('kappaEst1', tf.reshape(kappaEst1[1,:,:,0],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('kappaEst2', tf.reshape(kappaEst2[1,:,:,0],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('x_channel1', tf.nn.relu(tf.reshape(x_update[1,:,:,0],[1, imSize[1], imSize[2], 1]) ))
        tf.summary.image('x_channel2', tf.nn.relu(tf.reshape(x_update[1,:,:,1],[1, imSize[1], imSize[2], 1]) ))
        tf.summary.image('x_channel3', tf.nn.relu(tf.reshape(x_update[1,:,:,2],[1, imSize[1], imSize[2], 1]) ))
        tf.summary.image('x_channel4', tf.nn.relu(tf.reshape(x_update[1,:,:,3],[1, imSize[1], imSize[2], 1]) ))
    #    tf.summary.image('residual', (true - primal_result_pos)[..., zBatch // 2: zBatch // 2 + 1])
        tf.summary.image('true', tf.reshape(true[1],[1, imSize[1], imSize[2], 1]) )
        tf.summary.image('imag', tf.reshape(imag[1,:,:,0:4],[1, imSize[1], imSize[2], 4]) )
        tf.summary.image('diff', tf.reshape(true[1]-y_diff[1],[1, imSize[1], imSize[2], 1]) )
        
        merged_summary = tf.summary.merge_all()
        expName='DiffNet_' + fileOutName
        test_summary_writer, train_summary_writer = util.summary_writers(name, expName ,cleanup=False)
        
    
    print('session running')
    sess.run(tf.global_variables_initializer())
    
    feed_test={imag: dataDbar.test.images[0:bSize],
                 true: dataDbar.test.true[0:bSize]}
                 
    
    lVal=1.5e-3
    startIt=0
    for i in range(maxIter):
          
          batch = dataDbar.train.next_batch(bSize)
    #      test1 = batch[0]  
    #      test2 = batch[1]
          feed_train={imag: batch[0], true: batch[1], learningRate: lVal}
                 
          _, merged_summary_result_train = sess.run([train_step, merged_summary],
                                          feed_dict=feed_train)
          
          if i % 10000 == 0:
              lVal=lVal/2
          
          if i % 20 == 0:
            #train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            #testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
            
            it=i+startIt
            #print(len(dataDbar.test.images) - bSize - 1)
            tBeg = randint(tRand, len(dataDbar.test.images) - bSize - 1)
            #print(tBeg)
            #print(len(dataDbar.test.images) - bSize - 1)
            tEnd= tBeg+bSize
          
            feed_test={imag: dataDbar.test.images[tBeg:tEnd],
                         true: dataDbar.test.true[tBeg:tEnd]}
            
            test_accuracy, test_result = sess.run([loss, y_diff],
                                                          feed_dict=feed_test)
            
        
            loss_result, rel_loss_res, merged_summary_result = sess.run([loss, rel_loss, merged_summary],
                              feed_dict=feed_test)
        
            train_summary_writer.add_summary(merged_summary_result_train, it)
            test_summary_writer.add_summary(merged_summary_result, it)
        
            print('iter={}, loss={}, rel.loss={}'.format(i, loss_result,rel_loss_res))
    Kappa1,Kappa2,Kappa3,Kappa4,Kappa5,results = sess.run([Kappa1, Kappa2, Kappa3,Kappa4, Kappa5,x_sum],feed_dict = feed_test);  
    Kappa_list = []
    Kappa_list.append(Kappa1);
    Kappa_list.append(Kappa2);
    Kappa_list.append(Kappa3);
    Kappa_list.append(Kappa4);
    Kappa_list.append(Kappa5);
    dataDbar.test.images[tBeg:tEnd]
    dataDbar.test.true[tBeg:tEnd]                 
    save_path = saver.save(sess, filePath)
    print("Model saved in file: %s" % save_path)
    # get gamma and results
    dict_sio = {
            'Kappa':Kappa_list,
            'Input':dataDbar.test.images[tBeg:tEnd],
            'Output':results,
            'True':dataDbar.test.true[tBeg:tEnd]
            }
    sio.savemat(MatOutName, dict_sio)      
    print('Result Saved')

    
    
    sess.close();   
 #   tf.reset_default_graph()   

    print('--------------------> DONE <--------------------')

folder_tensorboard = '';
folder_data = '../processed4python/multiGamma/';       

# GAMMA merged
#def train():
dataSetTest  = folder_data+ 'DOCM4DiffNet_multiChannel_transl_10mm_mergedMus.mat'
netPath = 'netData/test_DOCM.ckpt'
logName = folder_tensorboard + 'test_DOCM_inv_oneChannel_tensordiff_lr1e-3_5layers_UNET_mergedMus'
dataSetTrain = folder_data+'DOCM4DiffNet_multiChannel_transl_10mm_mergedMus.mat'
savematName = 'RESULTS/'+'DOCM4DiffNet_multiChannel_UNET_transl_10mm_mergedMus.mat'
p = multiprocessing.Process(target=main, args=(netPath,logName,dataSetTrain,dataSetTest, 1300*3, savematName));
p.start()
p.join();
p.terminate()
print('Process done')
# GAMMA 1

dataSetTest  =folder_data+ 'DOCM4DiffNet_multiChannel_transl_10mm_liqMus3e-1.mat'
netPath = 'netData/test_DOCM.ckpt'
logName = folder_tensorboard +'test_DOCM_inv_oneChannel_tensordiff_lr1e-3_5layers_UNET_liqMus3e-1'
dataSetTrain = folder_data+'DOCM4DiffNet_multiChannel_transl_10mm_liqMus3e-1.mat'
savematName = 'RESULTS/'+'DOCM4DiffNet_multiChannel_UNET_transl_10mm_liqMus3e-1.mat'
p = multiprocessing.Process(target=main, args=(netPath,logName,dataSetTrain,dataSetTest, 1300,savematName));
p.start()
p.join()
p.terminate()
print('Process done')
# GAMMA 2

dataSetTest  = folder_data+  'DOCM4DiffNet_multiChannel_transl_10mm_liqMus0.45.mat'
netPath = 'netData/test_DOCM.ckpt'
logName = folder_tensorboard + 'test_DOCM_inv_oneChannel_tensordiff_lr1e-3_5layers_UNET_liqMus0.45'
dataSetTrain = folder_data+'DOCM4DiffNet_multiChannel_transl_10mm_liqMus0.45.mat'
savematName = 'RESULTS/'+'DOCM4DiffNet_multiChannel_UNET_transl_10mm_liqMus0.45.mat'
#main(netPath,logName,dataSetTrain,dataSetTest, 1300,savematName)
p = multiprocessing.Process(target=main, args=(netPath,logName,dataSetTrain,dataSetTest, 1300,savematName));
p.start()
p.join()
p.terminate()
print('Process done')
# GAMMA 3

dataSetTest  = folder_data+ 'DOCM4DiffNet_multiChannel_transl_10mm_liqMus0.6.mat'
netPath = 'netData/test_DOCM.ckpt'
logName = folder_tensorboard + 'test_DOCM_inv_oneChannel_tensordiff_lr1e-3_5layers_UNET_liqMus0.6'
dataSetTrain = folder_data+'DOCM4DiffNet_multiChannel_transl_10mm_liqMus0.6.mat'
savematName ='RESULTS/'+'DOCM4DiffNet_multiChannel_UNET_transl_10mm_liqMus0.6.mat'
p = multiprocessing.Process(target=main, args=(netPath,logName,dataSetTrain,dataSetTest, 1300,savematName));
p.start()
p.join()
p.terminate()
print('Process done')
#main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
