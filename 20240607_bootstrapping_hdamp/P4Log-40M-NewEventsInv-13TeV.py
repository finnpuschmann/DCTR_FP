#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
from tensorflow.keras.layers import Lambda, Dense, Input, Layer, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import concatenate
import tensorflow as tf
import argparse


gpus = tf.config.list_physical_devices('GPU')
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], 'GPU')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# standard numerical library imports
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import os

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, to_categorical

# Global plot settings
from matplotlib import rc
import matplotlib.font_manager
rc('font', family='serif')
rc('text', usetex=False)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batchsize', default=2048, type=int,
    help='specify the batchsize (default: %(default)s)'
)
parser.add_argument(
    '--evNum', default=40000000, type=int,
    help='specify the numEvents for the Train (default: %(default)s)'
)

args = parser.parse_args()
BatchSize = args.batchsize
evNum = args.evNum

print('BatchSize: '+str(BatchSize))
print('evNum: '+str(evNum))

outDir = '/afs/desy.de/user/v/vaguglie/www/NN/Train13TeV/BatchSize%s_NumEvents%s/InvertedGpu/' %(BatchSize, evNum)
if not os.path.isdir(outDir):
      os.makedirs(outDir)

def normPtHdamp(X):
    for i in range(0,len(X)):
        for l in range (0,2):
            for c in range(0,1):
                #if (X[i][l][c] != 0):
                X[i][l][c] = math.log10(X[i][l][c])
            for k in range(5,6):
                X[i][l][k] = 1.379


def normPtHdampTest(X):
    for i in range(0,len(X)):
        for l in range (0,2):
            for c in range(0,1):
                #if (X[i][l][c] != 0):
                X[i][l][c] = math.log10(X[i][l][c])
            for k in range(5,6):
                X[i][l][k] = 2.305

plot_style_0 = {'histtype':'step', 'color':'black', 'linewidth':2, 'linestyle':'--', 'density':True}
plot_style_1 = {'alpha':0.5, 'density':True}

def plotPt (X_0, X_1, title):

    plt.figure(figsize=(6,5))
    bins = np.linspace(0,5,20)
    hist0 = plt.hist(X_0[:,:,0], bins = bins, label = 'X0 gen (hdamp=2.305mtop)', **plot_style_1)
    hist1 = plt.hist(X_1[:,:,0], bins = bins, label = 'X1 gen (hdamp=1.379mtop)', **plot_style_1)

    plt.xlabel(r'$p_T(t\bar{t}) \; [GeV]$')
    plt.ylabel(r'$\dfrac{d\sigma}{dp_T(t\bar{t})}$')
    #plt.xlim([0,40])
    plt.legend()
    #plt.yscale('log')
    plt.savefig(outDir+'pTsys%s.pdf' %title)


def plotM (X_0, X_1, title):

    plt.figure(figsize=(6,5))
    bins = np.linspace(0,1,20)
    hist0 = plt.hist(X_0[:,:,3], bins = bins, label = 'X0 gen (hdamp=2.305mtop)', **plot_style_1)
    hist1 = plt.hist(X_1[:,:,3], bins = bins, label = 'X1 gen (hdamp=1.379mtop)', **plot_style_1)

    plt.xlabel(r'$m \; [GeV]$')
    plt.ylabel(r'$\dfrac{d\sigma}{dm}$')
    #plt.xlim([0,40])
    plt.legend()
    plt.yscale('log')
    plt.savefig(outDir+'massNorm%s.pdf' %title)


def plotID (X_0, X_1, title):

    plt.figure(figsize=(6,5))
    bins = np.linspace(0,1,20)
    hist0 = plt.hist(X_0[:,:,4], bins = bins, label = 'X0 gen (hdamp=2.305mtop)', **plot_style_1)
    hist1 = plt.hist(X_1[:,:,4], bins = bins, label = 'X1 gen (hdamp=1.379mtop)', **plot_style_1)

    plt.xlabel(r'$ID$')
    plt.ylabel(r'$\dfrac{d\sigma}{dID}$')
    #plt.xlim([0,40])
    plt.legend()
    plt.yscale('log')
    plt.savefig(outDir+'IDNorm%s.pdf' %title)


def plotHdamp (X_0, X_1, title):
    plot_style_0 = {'histtype':'step', 'color':'black', 'linewidth':2, 'linestyle':'--', 'density':True}
    plot_style_1 = {'alpha':0.5, 'density':True}

    plt.figure(figsize=(6,5))
    bins = np.linspace(0,5,20)
    hist0 = plt.hist(X_0[:,:,5], bins = bins, label = 'X0 gen (hdamp=2.305mtop)', **plot_style_1)
    hist1 = plt.hist(X_1[:,:,5], bins = bins, label = 'X1 gen (hdamp=1.379mtop)', **plot_style_1)

    plt.xlabel(r'$Hdamp$')
    plt.ylabel(r'$\dfrac{d\sigma}{dHdamp}$')
    #plt.xlim([0,40])
    plt.legend()
    plt.yscale('log')
    plt.savefig(outDir+'HdampNorm%s.pdf' %title)

def normalizeM(X0,X1):

    X = np.concatenate((X0,X1))
    maxM0 = max(X[:,0,3])
    maxM1 = max(X[:,1,3])
    maximum = max(maxM0, maxM1)
    print(maximum)
    ## norm mass
    for i in range(0,len(X0)):
        for l in range (0,2):
            for c in range(3,4):
                X0[i][l][c] =X0[i][l][c]/maximum

    for i in range(0,len(X1)):
        for l in range (0,2):
            for c in range(3,4):
                X1[i][l][c] =X1[i][l][c]/maximum

    return maximum


def normalizeMTest(X0,X1,maximum):

    print(maximum)
    ## norm mass
    for i in range(0,len(X0)):
        for l in range (0,2):
            for c in range(3,4):
                X0[i][l][c] =X0[i][l][c]/maximum

    for i in range(0,len(X1)):
        for l in range (0,2):
            for c in range(3,4):
                X1[i][l][c] =X1[i][l][c]/maximum


#seed = [1,2]
seed=[]
for i in range (1,51):
    seed.append(i)
print(seed)
num=seed

dataset=np.load('/nfs/dust/cms/user/vaguglie/converterLHEfiles/BaseNom13TeV/Results%i/2MeventsTrain_1.379_seed%i_Base1000_13TeV_P4.npz' %(seed[0], num[0]))
print(dataset.files)
X1=dataset['a']
print(X1.shape)

for i in range (1, len(seed)):
    dataset=np.load('/nfs/dust/cms/user/vaguglie/converterLHEfiles/BaseNom13TeV/Results%i/2MeventsTrain_1.379_seed%i_Base1000_13TeV_P4.npz' %(seed[i], num[i]))
    print(dataset.files)
    X=dataset['a']
    print(X.shape)
    X1 = np.concatenate((X1,X))
print(X1.shape)

dataset=np.load('/nfs/dust/cms/user/vaguglie/converterLHEfiles/BaseUp13TeV/Results%i/2MeventsTrain_2.305_seed%i_Base1000_13TeV_P4.npz' %(seed[0], num[0]))
print(dataset.files)
X0=dataset['a']
print(X0.shape)

for i in range (1, len(seed)):
    dataset=np.load('/nfs/dust/cms/user/vaguglie/converterLHEfiles/BaseUp13TeV/Results%i/2MeventsTrain_2.305_seed%i_Base1000_13TeV_P4.npz' %(seed[i], num[i]))
    print(dataset.files)
    X=dataset['a']
    print(X.shape)
    X0 = np.concatenate((X0,X))
print(X0.shape)

if (evNum==40000000):
    X0=X0
    X1=X1
else:
    X0 = X0[0:evNum,:,:]
    X1 = X1[0:evNum,:,:]
print(X0.shape)
print(X1.shape)


minimum = min (len(X0[:,0,0]),len(X1[:,0,0]))
print(minimum)

X0 = X0[0:minimum,:,:]
X1 = X1[0:minimum,:,:]

print(X0.shape)
print(X1.shape)

normPtHdamp(X0)
normPtHdamp(X1)

maximum = normalizeM(X0,X1)
print('maximumM: '+str(maximum))

X0 = np.delete(X0,2,1)
X1 = np.delete(X1,2,1)

plotPt(X0, X1, 'Train')
plotM(X0,X1, 'Train')

# PDGid to small float dictionary
PID2FLOAT_MAP_mine = {21: 0,
                 6: .1, -6: .2,
                 5: .3, -5: .4,
                 4: .5, -4: .6,
                 3: .7, -3: .8,
                 2: 0.9, -2: 1.0,
                 1: 1.1, -1: 1.2}

def remap_pids_mine(events, pid_i=4, error_on_unknown=True):
    """Remaps PDG id numbers to small floats for use in a neural network.
    `events` are modified in place and nothing is returned.
    **Arguments**
    - **events** : _numpy.ndarray_
        - The events as an array of arrays of particles.
    - **pid_i** : _int_
        - The column index corresponding to pid information in an event.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDG ID is
        encountered. If `False`, unknown PDG IDs will map to zero.
    """

    if events.ndim == 3:
        pids = events[:,:,pid_i].astype(int).reshape((events.shape[0]*events.shape[1]))
        if error_on_unknown:
            events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP_mine[pid]
                                            for pid in pids]).reshape(events.shape[:2])
        else:
            events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP_mine.get(pid, 0)
                                            for pid in pids]).reshape(events.shape[:2])
    else:
        if error_on_unknown:
            for event in events:
                event[:,pid_i] = np.asarray([PID2FLOAT_MAP_mine[pid]
                                             for pid in event[:,pid_i].astype(int)])
        else:
            for event in events:
                event[:,pid_i] = np.asarray([PID2FLOAT_MAP_mine.get(pid, 0)
                                             for pid in event[:,pid_i].astype(int)])

def preprocess_data(X):
    # Remap PIDs to unique values in range [0,1]
    remap_pids_mine(X, pid_i=4, error_on_unknown=True)
    return X

X0 = preprocess_data(X0)
X1 = preprocess_data(X1)

plotID(X0,X1,'Train')
plotHdamp(X0,X1,'Train')

Y0 = np.array([0. for i in range(X0.shape[0])])
Y1 = np.array([1. for i in range(X1.shape[0])])

print(Y0.shape)
print(Y1.shape)

Y = np.concatenate((Y0, Y1))
Y = to_categorical(Y, num_classes=2)

X = []
X = np.concatenate((X0,X1))
print(X.shape)
print(Y.shape)

X_train, X_val, Y_train, Y_val = data_split(X, Y, test=0.25, shuffle=True)

print(X_train.shape)
print(Y_train.shape)

print(X_val.shape)
print(Y_val.shape)

from math import *

# network architecture parameters
Phi_sizes = (100,100, 128)
F_sizes = (100,100, 100)

dctr = PFN(input_dim=6,
           Phi_sizes=Phi_sizes, F_sizes=F_sizes,
           summary=False)

from tensorflow import keras

save_label = 'DCTR_pp_tt_1D_hdamp1.379-2.305-P4-Log-%s-NewEvents-1000Epochs-Cut1000-batchSize%s-Inverted-GPU' %(evNum, BatchSize)

checkpoint = keras.callbacks.ModelCheckpoint('/afs/desy.de/user/v/vaguglie/saved_models_Hdamp/' + save_label + '.h5',
                                                monitor='val_loss',
                                                verbose=2,
                                                save_best_only=True,
                                                mode='min')

CSVLogger = keras.callbacks.CSVLogger('/afs/desy.de/user/v/vaguglie/logs/' + save_label + '_loss.csv', append=False)

EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=10,
                                              verbose=1,
                                              restore_best_weights=True)
callbacks = [checkpoint, CSVLogger, EarlyStopping]

print('Starting training')
history = dctr.fit(X_train, Y_train,
                    epochs = 1000,
                    batch_size = BatchSize,
                    validation_data = (X_val, Y_val),
                    verbose = 1,
                    callbacks = callbacks)

dctr.save('/afs/desy.de/user/v/vaguglie/saved_models_Hdamp/'+save_label+'.h5')

plt.figure(figsize=(6,5))
plt.plot(history.history['loss'],     label = 'loss')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.legend(loc=0)
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.savefig(outDir+'history.pdf')

