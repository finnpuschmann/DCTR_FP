#!/usr/bin/env python
# coding: utf-8

# import system modules
import sys
import os
import gc
import argparse

# import standard numerical modules
import numpy as np
import math

# import machine learning modules
import tensorflow as tf
import keras.backend as K

gpu = tf.config.list_physical_devices('GPU') # make sure GPU usage is enabled
print(gpu)

sys.path.append('../')
import DCTR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which Parallel Process is running?")
    # process_id arg
    parser.add_argument('--pid', '--process_id', 
        help="Int. Process ID for parallel computing. Default: 1", 
        type = int, default = 1)
    args = parser.parse_args()
    process_id = args.pid
else:
    process_id = 1



# # load data
# directory with pre converted lhe files as numpy arrays
data_dir = '../../Data' # modify as needed

# Load POWHEG hvq x0 datasets
# x0_nrm for training, x0_plt and x0_plt_nrm for calculating stats used to decide which model performs best
# only contain tt-pair; every event has order: 
    # tt-pair, top, anti-top
# every particle has arguments: 
    # [pt, y, phi, mass, eta, E, PID, w, theta]
    # [0 , 1, 2  , 3   , 4  , 5, 6  , 7, 8    ]

# POWHEG hvq
x0_nrm = []
x0_nrm = np.load(f'{data_dir}/POWHEG_hvq/showered/normed_lhe_01.npy')[:9543943] # 9543943 num of NNLO samples
print(f'POWHEG hvq x0_nrm.shape:     {x0_nrm.shape}')

# plotting data; different from training data; for calculating stats
x0_plt = []
x0_plt = np.load(f'{data_dir}/POWHEG_hvq/showered/converted_lhe_02.npy')[:9543943]
print(f'POWHEG hvq x0_plt.shape:     {x0_plt.shape}')

x0_plt_nrm = []
x0_plt_nrm = np.load(f'{data_dir}/POWHEG_hvq/showered/normed_lhe_02.npy')[:9543943]
print(f'POWHEG hvq x0_plt_nrm.shape: {x0_plt_nrm.shape}')


# MiNNLO x1
# training data
x1_nrm = []
x1_nrm = np.load(f'{data_dir}/MiNNLO/showered/normed_lhe.npy')
print(f'MiNNLO all particles x1_nrm.shape: {x1_nrm.shape}')

# plotting data
x1_plt = []
x1_plt = np.load(f'{data_dir}/MiNNLO/showered/converted_lhe.npy')
print(f'MiNNLO all particles x1_plt.shape: {x1_plt.shape}')

# get normalized event generator weights | all weigths = +/-1
x0_wgt = x0_nrm[:, 0, 7].copy()
x0_plt_wgt = x0_plt_nrm[:, 0, 7].copy() 

x1_wgt = x1_nrm[:, 0, 7].copy()
x1_plt_wgt = x1_plt[:, 0, 7].copy()



# delete eta (pseudorapidity) and Energy -> Train only with [pt, y, phi, m, PID]

# delete energy
x0_nrm = np.delete(x0_nrm, 5, -1)
x0_plt_nrm = np.delete(x0_plt_nrm, 5, -1)
x1_nrm = np.delete(x1_nrm, 5, -1)

# delete eta
x0_nrm = np.delete(x0_nrm, 4, -1)
x0_plt_nrm = np.delete(x0_plt_nrm, 4, -1)
x1_nrm = np.delete(x1_nrm, 4, -1)


# prep arrays for training
x_train, x_val, y_train, y_val, wgt_train, wgt_val = DCTR.prep_arrays(x0_nrm, x1_nrm, val=0.25)


K.clear_session()
gc.collect() # cpu gabage collection to free up memory from discarded temp arrays


train_dir = './saved_models' # where to save models during training

os.makedirs(train_dir, exist_ok=True)

# MSE loss
# setup DCTR neural network
''' default setup_nn() args:
input_dim=5, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', reduceLR = True,
mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False, verbose = 2
'''

dctr, callbacks = DCTR.setup_nn(saveLabel=f'20240903_DCTR_training_mse_{process_id}', savePath=f'{train_dir}/', loss='mse', patience=30, learning_rate=0.001, use_scheduler=False)



# train DCTR neural network
''' train() necessary args:
dctr, callbacks, X_train, Y_train, X_val, Y_val
default args: 
wgt_train=1.0, wgt_val=1.0, epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training', verbose = 2, plot=True
'''


DCTR.train(dctr, callbacks, x_train, y_train, x_val, y_val, wgt_train=wgt_train, wgt_val=wgt_val, batch_size=8192*16, saveLabel=f'20240903_DCTR_training_mse_{process_id}', savePath=f'{train_dir}/', epochs=250)


# CCE loss
gc.collect()
K.clear_session()

# setup DCTR neural network
''' default setup_nn() args:
input_dim=5, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', reduceLR = True,
mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False, verbose = 2
'''

dctr, callbacks = DCTR.setup_nn(saveLabel=f'20240903_DCTR_training_cce_{process_id}', savePath=f'{train_dir}/', loss='cce', patience=30, learning_rate=0.001, use_scheduler=False)


# train DCTR neural network
''' train() necessary args:
dctr, callbacks, X_train, Y_train, X_val, Y_val
default args: 
wgt_train=1.0, wgt_val=1.0, epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training', verbose = 2, plot=True
'''

DCTR.train(dctr, callbacks, x_train, y_train, x_val, y_val, wgt_train=wgt_train, wgt_val=wgt_val, batch_size=8192*16, saveLabel=f'20240903_DCTR_training_cce_{process_id}', savePath=f'{train_dir}/', epochs=250)
