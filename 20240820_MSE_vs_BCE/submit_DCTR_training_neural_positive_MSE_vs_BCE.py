import sys
import argparse
import gc
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

print('GPUs available for tensorflow:')
print(tf.config.list_physical_devices('GPU'))

sys.path.append('../')
import DCTR

from energyflow.utils import to_categorical
from sklearn.model_selection import train_test_split


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


K.clear_session()
gc.collect() # cpu gabage collection to free up memory from discarded temp arrays



# neural positive reweigher for MiNNLO X1
x = np.concatenate([x1_nrm, x1_nrm]) # two identical copies
wgt = np.concatenate([x1_wgt, np.ones(len(x1_wgt))]) # wgts as is and all = 1

y = np.concatenate([np.ones(len(x1_nrm)), np.zeros(len(x1_nrm))]) # identifier
y = to_categorical(y, num_classes=2) # encode y as one hot

# split x, y, wgt into random training and validation datasets
X_rwgt_train, X_rwgt_val, Y_rwgt_train, Y_rwgt_val, wgt_rwgt_train, wgt_rwgt_val = train_test_split(x[...,:-2].copy(), y.copy(), wgt.copy(), test_size=0.15)

# clear tmp arrays from memory
del x
del y
del wgt

K.clear_session()
gc.collect() # cpu gabage collection to free up memory from discarded temp arrays


# neural positive training:
# compare MSE vs CCE neural positive training
# repeat each training 5 times to get a mean and std for each method and batch size
# try batch sizes of: [4, 8, 16, 32, 64]*8192 for MSE and CCE => 5*5*2=50 trainings
# -> split to run parallel on NAF
# first 25 with MSE, last 25 with CCE

# process_id between 1 and 50

runs = 5
batch_sizes = [4, 8, 16, 32, 64] # mulipliers of 8192

total_runs = 2 * runs * len(batch_sizes) # *2 b/c testing mse and cce

run = (process_id % runs) # process_id starts at 1
if run == 0:
    run = runs # last run would be zero with modulo

loss = ''
if process_id <= int(0.5*total_runs):
    loss = 'mse'
else:
    loss = 'cce'

# want the first num runs to use batch_sizes[0], next num runs batch_sizes[1], etc.
# split process_id into two sets of size int(0.5*total_runs), one for each loss
if process_id <= int(0.5*total_runs):
    p_id = process_id # 1-15 for MSE
else:
    p_id = process_id - int(0.5*total_runs) # 1-15 for CCE

batch_id = int((p_id-1)/runs)
batch_size = 8192*batch_sizes[batch_id]


print('setting up neural network and starting training')

# set up NN and CallBacks
# defaults:
'''
input_dim=5, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', reduceLR = True,
mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False, verbose = 2
'''

pos_rwgt, cb = DCTR.setup_nn(
    patience=15,
    loss = loss,
    use_scheduler=False,
    reduceLR=True,
    saveLabel=f'DCTR_NNLO_{loss}_pos_rwgt_{run}_batchsize_{batch_size}',
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
    pos_rwgt,
    cb,
    X_rwgt_train,
    Y_rwgt_train,
    X_rwgt_val,
    Y_rwgt_val,
    wgt_rwgt_train,
    wgt_rwgt_val,
    batch_size=batch_size,
    epochs=100,
    saveLabel=f'DCTR_NNLO_{loss}_pos_rwgt_{run}_batchsize_{batch_size}',
    savePath='./saved_models/',
)

print(f'finished training: {loss = }, {run = }, {batch_size = }')
