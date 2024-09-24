import sys
import argparse
import itertools
import gc

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

print('GPUs available for tensorflow:')
print(tf.config.list_physical_devices('GPU'))

sys.path.append('../')
import DCTR

from energyflow.utils import to_categorical
from energyflow.utils import data_split


# parse cli arguments
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
print('loading data')
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
# print(f'POWHEG hvq x0_nrm.shape:     {x0_nrm.shape}')

# plotting data; different from training data; for calculating stats
x0_plt = []
x0_plt = np.load(f'{data_dir}/POWHEG_hvq/showered/converted_lhe_02.npy')[:9543943]
# print(f'POWHEG hvq x0_plt.shape:     {x0_plt.shape}')

x0_plt_nrm = []
x0_plt_nrm = np.load(f'{data_dir}/POWHEG_hvq/showered/normed_lhe_02.npy')[:9543943]
# print(f'POWHEG hvq x0_plt_nrm.shape: {x0_plt_nrm.shape}')


# MiNNLO x1
# training data
x1_nrm = []
x1_nrm = np.load(f'{data_dir}/MiNNLO/showered/normed_lhe.npy')
# print(f'MiNNLO all particles x1_nrm.shape: {x1_nrm.shape}')

# plotting data
x1_plt = []
x1_plt = np.load(f'{data_dir}/MiNNLO/showered/converted_lhe.npy')
# print(f'MiNNLO all particles x1_plt.shape: {x1_plt.shape}')

# get normalized event generator weights | all weigths = +/-1
x0_wgt = x0_nrm[:, 0, 7].copy()
x0_plt_wgt = x0_plt_nrm[:, 0, 7].copy() 

x1_wgt = x1_nrm[:, 0, 7].copy()
x1_plt_wgt = x1_plt[:, 0, 7].copy()


# prep data
print('preparing data')
# delete eta (pseudorapidity) and Energy -> Train only with [pt, y, phi, m, PID]

# delete energy
x0_nrm = np.delete(x0_nrm, 5, -1)
x0_plt_nrm = np.delete(x0_plt_nrm, 5, -1)
x1_nrm = np.delete(x1_nrm, 5, -1)

# delete eta
x0_nrm = np.delete(x0_nrm, 4, -1)
x0_plt_nrm = np.delete(x0_plt_nrm, 4, -1)
x1_nrm = np.delete(x1_nrm, 4, -1)


# load rwgt from neural positive
# best results:
    # MSE: 4x8192, run (index) 1 -> 2
    # BCE: 4x8192, run (index) 4 -> 5
batch_size = 4*8192
save_path = './saved_models/'
mse_label = f'DCTR_NNLO_mse_pos_rwgt_2_batchsize_{batch_size}'
cce_label = f'DCTR_NNLO_cce_pos_rwgt_5_batchsize_{batch_size}'

x1_rwgt_mse = np.load(f'{save_path}{mse_label}_X1_rwgt.npy')
x1_rwgt_cce = np.load(f'{save_path}{cce_label}_X1_rwgt.npy')


K.clear_session()
gc.collect() # cpu gabage collection to free up memory from discarded temp arrays


# prep arrays for training
x = np.concatenate((x0_nrm[...,:-2], x1_nrm[...,:-2]))
wgts_org = np.concatenate((x0_wgt, x1_wgt))
wgts_mse = np.concatenate((x0_wgt, x1_rwgt_mse))
wgts_cce = np.concatenate((x0_wgt, x1_rwgt_cce))

y0 = x0_nrm[:,0,-1].copy() # theta is last parameter
y1 = x1_nrm[:,0,-1].copy()
y = np.concatenate((y0, y1))
y = to_categorical(y, num_classes=2)

x_train, x_val, y_train, y_val, wgt_train_org, wgt_val_org, wgt_train_mse, wgt_val_mse, wgt_train_cce, wgt_val_cce  = data_split(x, y, wgts_org, wgts_mse, wgts_cce, test=0.25)

with tf.device('/cpu:0'):
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_val = tf.convert_to_tensor(x_val)
    y_val = tf.convert_to_tensor(y_val)


K.clear_session()
gc.collect() # cpu gabage collection to free up memory from discarded temp arrays

# compare neural positive MSE vs CCE results to org weights
# try both CCE and MSE losses for each weight
# repeat each training 5 times to get a mean and std for each method and batch size
# try batch sizes of: [4, 8, 16, 32]*8192 for MSE and CCE => 2*3*4*5=24*5 trainings
# run all 5 runs for each set on one machine

runs = 5
losses = ['mse', 'cce']
batch_sizes = [4, 8, 16, 32] # mulipliers of 8192
wgts = ['org', 'np_mse', 'np_cce']

combinations = list(itertools.product(losses, batch_sizes, wgts))

loss, batch_mult, wgt_label = combinations[process_id - 1]
batch_size = batch_mult*8192
if wgt_label == 'org':
    wgt_train = wgt_train_org
    wgt_val = wgt_val_org
elif wgt_label == 'np_mse':
    wgt_train = wgt_train_mse
    wgt_val = wgt_val_mse
else:
    wgt_train = wgt_train_cce
    wgt_val = wgt_val_cce

# set up NN and CallBacks
# defaults:
'''
input_dim=5, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', reduceLR = True,
mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False, verbose = 2
'''
for run in range(1, runs+1):
    dctr_nn, cb = DCTR.setup_nn(
        patience=15,
        loss = loss,
        use_scheduler=False,
        reduceLR=True,
        saveLabel=f'DCTR_NNLO_wgt_{wgt_label}_loss_{loss}_{run}_batchsize_{batch_size}',
        savePath='./saved_models/',
        summary=False
    )
    
    # train NN
    '''
    neecessary args:
    dctr, callbacks, X_train, Y_train, X_val, Y_val, 
    
    default args:
    wgt_train=1.0, wgt_val=1.0, epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training', verbose = 2, plot=True):
    '''
    
    DCTR.train(
        dctr_nn,
        cb,
        x_train,
        y_train,
        x_val,
        y_val,
        wgt_train,
        wgt_val,
        batch_size=batch_size,
        epochs=100,
        saveLabel=f'DCTR_NNLO_wgt_{wgt_label}_loss_{loss}_{run}_batchsize_{batch_size}',
        savePath='./saved_models/',
    )
    
    print(f'finished training: {loss = }, {run = }, {batch_size = }')
