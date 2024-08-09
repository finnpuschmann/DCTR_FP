#!/usr/bin/env python

# standard library imports
from __future__ import absolute_import, division, print_function
import os
import sys
import argparse
import gc

# standard numerical library imports
import numpy as np
import scipy as sp

# tensorflow and keras imports
import tensorflow as tf
from tensorflow import keras

import keras.backend as K
from tensorflow.keras.layers import Lambda, Dense, Input, Layer, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import concatenate

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, remap_pids, to_categorical


# parse cli arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="How many training iterations shoud be run?")
    # iter arg
    parser.add_argument("-i", "--iter", help="Int. Number of repeated training runs with randomized samples", default = 5)
    # prcocess_id arg
    parser.add_argument("-p", "--process", help="Int. ID of Which process is running, when running multiple in parallel", default = 1)
    args = parser.parse_args()
    ITER = args.iter
    PROCESS = args.process
else:
    ITER = 5
    PROCESS = 1

memory=8192*0.9
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

# define training iter function and helpful setup_nn function
def setup_nn(Phi_sizes = (100,100,128), F_sizes = (100,100,100), input_dim=1, patience = 15, save_label = 'DCTR_pp_tt_1D_Rb_mine_xB_CP5_nominal', out_dir = './saved_models'):

    dctr = PFN(input_dim = input_dim,
               Phi_sizes = Phi_sizes, 
               F_sizes   = F_sizes,
               summary   = False)

    os.makedirs(os.path.dirname(f'{out_dir}/{save_label}.h5'), exist_ok=True) # create output dir, if it doesn't exist
    
    checkpoint = keras.callbacks.ModelCheckpoint(f'{out_dir}/{save_label}.h5',
                                                    monitor='val_loss',
                                                    verbose=2,
                                                    save_best_only=True,
                                                    mode='min')
    
    CSVLogger = keras.callbacks.CSVLogger(f'{out_dir}/{save_label}_loss.csv', append=False)
    
    EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=patience,
                                                  verbose=1,
                                                  restore_best_weights=True)
    
    callbacks = [checkpoint, CSVLogger, EarlyStopping]

    return dctr, callbacks


def train_single_iteration(X0, X1, iteration, num_events = int(4e6), batch_size = 1000, save_label = 'DCTR_pp_tt_1D_Rb_mine_xB_CP5_nominal', out_dir = './saved_models'):
    K.clear_session()
    gc.collect() # collect garbage to free memory
    
    # take random num_evts from each dataset
    rand0 = np.random.choice(len(X0), num_events) # num_evts random indices
    rand1 = np.random.choice(len(X1), num_events)

    local_X0 = X0[rand0]
    local_X1 = X1[rand1]

    # create classifier array Y
    Y0 = np.array([0. for i in range(local_X0.shape[0])])
    Y1 = np.array([1. for i in range(local_X1.shape[0])])

    Y = np.concatenate((Y0, Y1))
    Y = to_categorical(Y, num_classes=2)

    # create training array
    X = []
    X = np.concatenate((local_X0, local_X1))
    
    X_train, X_val, Y_train, Y_val = data_split(X, Y, test=0.25, shuffle=True)

    del rand0, rand1, local_X0, local_X1, X, Y # delete the tmp variables to free memory
    gc.collect() # collect garbage to free memory

    with tf.device('/cpu:0'):
        X_train = tf.convert_to_tensor(X_train)
        X_val   = tf.convert_to_tensor(X_val)
        Y_train = tf.convert_to_tensor(Y_train)
        Y_val   = tf.convert_to_tensor(Y_val)
    
    # ready to start training
    dctr, callbacks = setup_nn(save_label = f'{save_label}_iter_{iteration:02d}', out_dir = out_dir)

    print('Starting training')
    history = dctr.fit(X_train, Y_train,
                       epochs = 1000,
                       batch_size = batch_size,
                       validation_data = (X_val, Y_val),
                       verbose = 1,
                       callbacks = callbacks)
    
    dctr.save(f'{out_dir}/{save_label}_iter_{iteration:02d}.h5')
    
    plt.figure(figsize=(6,5))
    plt.plot(history.history['loss'],     label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val loss')
    plt.legend(loc=0)
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.savefig(f'{out_dir}/{save_label}_iter_{iteration:02d}_history.pdf')

    del history, dctr, callbacks, X_train, X_val, Y_train, Y_val # delete vars to free memory
    K.clear_session()
    gc.collect() # collect garbage to free memory


# local docker
# data_dir = '../../Data'

# NAF
data_dir = '/nfs/dust/cms/user/puschman/pythia8307/examples/output'


# X0: Rb 1.056
# X1: Rb 0.855

# load data

X0 = []
for i in range(1, 13):
    dataset = np.load(f'{data_dir}/B-Fragmentation_Rb_1.056/bootstrapping_Xb_multC_multNeutra_listBtop_listBextra-Rb_1.056_1M_seed{i}_CP5.npz')
    # print(dataset.files)
    X0.extend(dataset['a'])

X0 = np.array(X0)
# print(X0.shape)

X1 = []
for i in range(1, 13):
    dataset = np.load(f'{data_dir}/B-Fragmentation_Rb_0.855/bootstrapping_Xb_multC_multNeutra_listBtop_listBextra-Rb_0.855_1M_seed{i}_CP5.npz')
    # print(dataset.files)
    X1.extend(dataset['a'])

X1 = np.array(X1)
# print(X1.shape)

# process data
X0_pari = []
X0_dispari = []
X1_pari = []
X1_dispari = []


for i, _ in enumerate(X0):
    if i % 2 == 0:
        X0_pari.append(X0[i])
    else:
        X0_dispari.append(X0[i])


for i, _ in enumerate(X1):
    if i % 2 == 0:
        X1_pari.append(X1[i])
    else:
        X1_dispari.append(X1[i])


X0_tot = []
for i, _ in enumerate(X0_pari):
    X0_tot.append([[X0_pari[i]], [X0_dispari[i]]])
X0_tot = np.array(X0_tot)


X1_tot = []
for i, _ in enumerate(X1_pari):
    X1_tot.append([[X1_pari[i]], [X1_dispari[i]]])
X1_tot = np.array(X1_tot)


# TRAINING

# NUM iterations with 4M random samples per iter

for i in range(PROCESS, PROCESS + ITER + 1):
    train_single_iteration(X0_tot, X1_tot, iteration=i)

