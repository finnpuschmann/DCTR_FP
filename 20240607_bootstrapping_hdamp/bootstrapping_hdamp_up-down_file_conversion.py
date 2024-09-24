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
from tensorflow import keras
import tensorflow.keras.backend as K


import gc
import argparse



# standard numerical library imports
import numpy as np
import math
from math import *
import scipy as sp
import matplotlib.pyplot as plt
import os

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, to_categorical



# reverse engineering normalization function
# pt -> np.log10(pt) ?
# rapidity and phi aren't touched
# mass = mass / max(mass) ?
# pid: top:  6->0.1
#     atop: -6->0.2

# hdamp_val: 1.379

def normalize(X):
    X[:,0,0] = np.log10(X[:,0,0]) # log pt
    X[:,1,0] = np.log10(X[:,1,0])
    
    X[:,0,3] = X[:,0,3]/np.max(X[:,0,3]) # mass
    X[:,1,3] = X[:,1,3]/np.max(X[:,1,3]) 
    
    X[:,0,4] = 0.1 # pid
    X[:,1,4] = 0.2
    
    X[:,0,5] = 1.379 # hdamp
    X[:,1,5] = 1.379

    return X



# convert datasets to normed

# data_dir     = '/tf/data/BachelorThesis_Data/Valentinas_Samples'
sample_dir = '/nfs/dust/cms/user/vaguglie/converterLHEfiles/'

# where to save to
data_dir = '../../Data/'

for i in range(1,51):
    os.makedirs(f'{data_dir}/hdamp/down/Results{i}', exist_ok=True)
    os.makedirs(f'{data_dir}/hdamp/up/Results{i}', exist_ok=True)
    os.makedirs(f'{data_dir}/hdamp/nominal/Results{i}', exist_ok=True)

# X0: down
# X1: nominal

# shape of the datasets in valentinas script was (:, 2, 6)
# shape of the datasets passed to me are         (:, 3, 6)
# so removing 3. entry to be left with only top and anti-top

print('begining file conversion...')

for i in range(1, 51):
    X0 = [] # down
    X1 = [] # up
    X2 = [] # nominal
    X0_nrm = []
    X1_nrm = []
    X2_nrm = []

    
    X0 = np.load(f'{sample_dir}/BaseDown13TeV/Results{i}/2MeventsTrain_0.8738_seed{i}_Base1000_13TeV_P4.npz')['a'][:,:-1,:]    
    X0_nrm = np.array(normalize(X0))
    np.save(f'{data_dir}/hdamp/down/Results{i}/2MeventsTrain_0.8738_seed{i}_Base1000_13TeV_P4_normed.npy', X0_nrm)


    X1 = np.load(f'{sample_dir}/BaseUp13TeV/Results{i}/2MeventsTrain_2.305_seed{i}_Base1000_13TeV_P4.npz')['a'][:,:-1,:]    
    X1_nrm = np.array(normalize(X1))
    np.save(f'{data_dir}/hdamp/up/Results{i}/2MeventsTrain_2.305_seed{i}_Base1000_13TeV_P4_normed.npy', X1_nrm)


    X2 = np.load(f'{sample_dir}/BaseNom13TeV/Results{i}/2MeventsTrain_1.379_seed{i}_Base1000_13TeV_P4.npz')['a'][:,:-1,:]    
    X2_nrm = np.array(normalize(X2))
    np.save(f'{data_dir}/hdamp/nominal/Results{i}/2MeventsTrain_1.379_seed{i}_Base1000_13TeV_P4_normed.npy', X2_nrm)

    del X0, X1, X2, X0_nrm, X1_nrm, X2_nrm
    gc.collect()
    
    print(f'loaded and normalized nominal, down, and up datasets: \n\
        up to Results{i}')
