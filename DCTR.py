#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: finn puschmann
"""


# energyflow dependencies import
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import keras.backend as K

# system modules
import sys
import os
import glob
import math 
import multiprocessing as mp
import gc

# standard numerical library imports
import numpy as np
from scipy import stats
from math import atan2
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
import csv
from numba import cuda # for memory management

# energyflow imports
from energyflow.archs import PFN
from energyflow.utils import data_split, to_categorical

# madgraph imports
sys.path.append('/tf/madgraph/MG5_aMC_v2_9_16') # inside docker
sys.path.append('/home/finn/madgraph/MG5_aMC_v2_9_16') # running locally
try:
    from madgraph.various.lhe_parser import FourMomentum, EventFile
except ModuleNotFoundError:
    print('Madgraph was not found in PATH or in docker /tf/madgraph/MG5_aMC_v2_9_16 dir \n can be added temporarily with sys.path.append(\'path/to/madgraph\')')
 
# create variable of current working directory
currentPath = str(os.getcwd()+'/') # used as default training save/load dir


#################################################################################
'''
LHE Data Processing
Functions and methods used to convert lhe files to nump arrays and save them to disk as compressed .npz
Adds tt-pair, followed by top, anti-top and any number of jet particles to an array and saves it

'''
#################################################################################


def phi(particle): # calculates the angle phi of a particle from it's fourMomentum
    return atan2(particle.py, particle.px)


def pseudorapidity(particle):
    pz = particle.pz
    p = np.sqrt(np.power(particle.px, 2) + np.power(particle.py, 2) + np.power(particle.pz, 2))
    if (p-pz) == 0.0:
        raise Exception("Error calculating pseudorapidity (divide by zero)")
    elif ((p+pz)/(p-pz)) <= 0.0:
        raise Exception("Error calculating pseudorapidity (log of negative number)")
    else:
        pseudorapidity = 0.5*math.log((p+pz)/(p-pz))
        return pseudorapidity



def delta_phi(X):
    phi_t = X[:,1, 2] # phi of top quark
    phi_t_bar = X[:,2, 2] # phi of anti-top quark
    delta = phi_t - phi_t_bar
    # remap delta from [-2*pi, 2*pi] to [0, 2pi]
    delta = ((delta + 2*math.pi) % (2*math.pi))
    # remap to [0, pi] since it is symmetrical
    delta = math.pi - abs(delta-math.pi)
    
    return delta



def check_pseudorapidity(particle):
    pz = particle.pz
    p = np.sqrt(np.power(particle.px, 2) + np.power(particle.py, 2) + np.power(particle.pz, 2))
    if (p-pz) == 0.0:
        return False
    elif ((p+pz)/(p-pz)) <= 0.0:
        return False
    else: 
        return True


def check_rapidity(particle):
    '''
    checks if calculating rapidity would give you a calculation error. 
    returns True if calculating is no problem, returns False otherwise
    '''
    if (particle.E - particle.pz) == 0: # can't have a devide by zero
        return False
    if (particle.E + particle.pz)/(particle.E - particle.pz) <= 0: # the log of this is calculated for rapidity; can't take the log of zero or a negative number
        return False
    else: return True


def process_event(event, maxJetParts, theta, double_jet=False):
    '''
    is called by process_file for every event in its LHE file
    inherits filename, maxJetParts and theta from process_file and thus from convertLHE
    
    goes through an event particle by particle, adding the top, anti-top,
    as well as up to maxJetParts other quarks and gluons in an event to arrays.
    Calculates some properties for the particles, like phi and pseudorapidity and discards
    some particles and events if it comes to domainErrors.
    
    returns the eventVector with all particles including tt-pair of the event
    '''
    w = event.wgt
    ptop = FourMomentum(0, 0, 0, 0)
    pantitop = FourMomentum(0, 0, 0, 0)
    pjet = FourMomentum(0, 0, 0, 0)
    top_eta = 0
    antitop_eta = 0
    jet_eta = 0
    eventVector = []
    eventJetVector = []
    countRapidity = 0
    countEta = 0
    
    # particle processing
    for particle in event:  # loops through every particle in event
        try:
            # top and anti-top
            if particle.status == 2:  # particle.status = 2: unstable particles, here only the top or anti-top are saved, W Bosons are ignored
                if particle.pid == 6:  # top quark
                    if check_rapidity(particle) == True:  # if rapidity is fine
                        try:
                            top_eta = pseudorapidity(particle)
                            if top_eta <= 1e6:
                                ptop = FourMomentum(particle)  # create FourMomentum for top
                            else:
                                print('top 4-moment error')
                                ptop = None
                                countEta += 1
                                continue
                        except:
                            print('top 4-moment error')
                            ptop = None
                            countEta += 1
                            continue
                    else:
                        print('top 4-moment error')
                        ptop = None
                        countRapidity += 1
                        continue

                elif particle.pid == -6:  # anti-top quark
                    if check_rapidity(particle) == True:  # if rapidity is fine
                        try:
                            antitop_eta = pseudorapidity(particle)
                            if antitop_eta <= 1e6:
                                pantitop = FourMomentum(particle)  # create FourMomentum for anti-top
                            else:
                                print('antitop 4-moment error')
                                pantitop = None
                                countEta += 1
                                continue
                        except:
                            print('antitop 4-moment error')
                            pantitop = None
                            countEta += 1
                            continue

                    else:
                        print('antitop 4-moment error')
                        pantitop = None
                        countRapidity += 1
                        continue

                else:
                    continue

            # jet particles
            jet_eta = 0
            if len(eventJetVector) < maxJetParts:  # limit to maxJetParts particles per event,
                # to avoid ragged arrays, if the event has fewer particles than the rest is filled with zeros
                if particle.status == 1:  # particle.status = 1: stable particles, here only jet particles (quarks and gluons) considered
                    if ((particle.pid < 6 and particle.pid > -6) or particle.pid == 21):  # only quarks and gluons are saved
                        if check_rapidity(particle) == True:  # if rapidity is fine
                            if check_pseudorapidity(particle) == True:
                                jet_eta = pseudorapidity(particle)
                                if jet_eta <= 1e6:
                                    pjet = FourMomentum(particle)  # FourMomentum of particle in Jet
                                    double_jet_prob = 0.6907047702952649  # number of two quark pairs/number of at least 1 quark in all MiNNLO Datasets
                                    if (double_jet == True and np.random.uniform() <= double_jet_prob):
                                        rng_split = np.clip(np.random.normal(0.5, scale=0.01), 0, 1)
                                        part_0 = rng_split * pjet
                                        part_1 = (1 - rng_split) * pjet
                                        while (check_pseudorapidity(part_0) == False or check_pseudorapidity(part_1) == False):  # redo split until correct pseudorapidities are created
                                            rng_split = np.clip(np.random.normal(0.5, scale=0.01), 0, 1)
                                            part_0 = rng_split * pjet
                                            part_1 = (1 - rng_split) * pjet

                                        eta0 = pseudorapidity(part_0)
                                        eta1 = pseudorapidity(part_1)
                                        eventJetVector.append(
                                            [part_0.pt, part_0.rapidity, phi(part_0), part_0.mass, eta0, part_0.E,
                                             particle.pid, w, theta])  # add particle to Jet Vector of event
                                        eventJetVector.append(
                                            [part_1.pt, part_1.rapidity, phi(part_1), part_1.mass, eta1, part_1.E,
                                             particle.pid, w, theta])  # add particle to Jet Vector of event

                                    else:
                                        eventJetVector.append(
                                            [pjet.pt, pjet.rapidity, phi(pjet), pjet.mass, jet_eta, pjet.E, particle.pid,
                                             w, theta])  # add particle to Jet Vector of event

                                else:
                                    countEta += 1
                                    continue
                            else:
                                countEta += 1
                                continue
                        else:
                            countRapidity += 1
                            continue

                    else:
                        continue

                else:
                    continue

            else:
                continue

        except:
            continue

    # sort eventJetVector so that Gluons come first, followed by the heavier quarks and ends with lightest quarks. -> decreasing absolute value of PID (arg 6)
    eventJetVector.sort(key=lambda x: (abs(x[6]), x[6]), reverse=True)

    # check if top or antitop 4-momentum is set to None -> Error in pseudorpaidity
    if (ptop is not None) and (pantitop is not None):
        p_tt = ptop + pantitop  # create madgraph FourMomentum of tt-pair
        # Top pair processing
        try:  # pseudorapidity for tt-pair
            tt_eta = pseudorapidity(p_tt)

            # for each event: 1. tt-pair, 2. top, 3. anti-top, followed by maxJetParts of jet particles
            eventVector.append(
                [p_tt.pt, p_tt.rapidity, phi(p_tt), p_tt.mass, tt_eta, p_tt.E, 0, w, theta])  # add tt-pair to output array
            eventVector.append(
                [ptop.pt, ptop.rapidity, phi(ptop), ptop.mass, top_eta, ptop.E, 6, w, theta])  # add top quark to event vector
            eventVector.append(
                [pantitop.pt, pantitop.rapidity, phi(pantitop), pantitop.mass, antitop_eta, pantitop.E, -6, w, theta])  # add anti-top quark to event vector
            
            # make sure eventJetVector has length of maxJetParts:
            eventJetVector = eventJetVector[:maxJetParts] + [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * (maxJetParts - len(eventJetVector))
            eventVector.extend(eventJetVector)
            
            return eventVector, countRapidity, countEta

        except:
            countEta += 1
    
    return None, None, countEta

def process_file(filename, maxJetParts, theta, double_jet = False):
    '''
    is called by convertLHE for every LHE file in it's inputFolder
    inherits filename, maxJetParts and theta from convertLHE
    
    opens lhe files and goes through it event by event, 
    calling the process_event(event, maxJetParts, theta) function 
    and returns a arrays of all events in the lhe file.
    
    returns an array each for all quarks and gluons and for only the tt-pair
        also returns a count of how many particles and events were skipped due to domain_errors in calculating one of their properties
    '''
    
    lhe1 = EventFile(filename) # uses madgraphs EventFile function to open the lhe file
    print(f'opening file: {filename}')
    lheVector = []
    countEta = 0
    countRapidity = 0

    for event in lhe1: # goes through the lhe file, event by event
        eventVector, rapidity, eta = process_event(event, maxJetParts, theta, double_jet ) # calls the process_event function for every event in lhe file. 
        countEta += eta
        if eventVector is not None: # append current event vectors to the rest of the event vectors of the lhe file
            lheVector.append(eventVector)
            countRapidity += rapidity

    return lheVector, countRapidity, countEta


def process_file_wrapper(args):
    '''
    wrapper for multiprocessing inside convert_lhe()
    simply calls the process_file() function
    '''
    filename, maxJetParts, theta, double_jet = args
    return process_file(filename, maxJetParts, theta, double_jet)


def convert_lhe(inputFolder, outputFolder, theta, outLabel=None, maxJetParts=8, label='converted_lhe', recursive=True, double_jet = False):
    '''
    main method to call to convert a bunch of lhe files in the inputFolder directory to two numpy arrays and save them to disk.
    
    Goes through inputFolder and adds all LHE files to the list lista, also checks subfolders when recursive=True
    Using multiprocessing, calls process_file for every lhe file and append their all_particles and tt-pair arrays.
    saves the resulting numpy arrays to disk with compression.
    
    all_particles array shape: (numEvents,particlesPerEvent, numArgsPerParticle)
    with args [p_T, rapidity, phi, mass, PID, eventWeight, theta]

    tt-pair array shape: (numEvents, numArgsPerTT-pair)
    with args [p_T, rapidity, phi, mass, pseudorapidity, eventWeight, Energy, theta]
    
    Argumets:
    inputFolder: path to directory containing the lhe files to be converted
    
    outputFolder: path to directory where numpy arrays are to be saved on disk
    
    theta: classification paramter that the Neural Network is trained to learn; typically: theta = 0 for POWHEG_hvq and theta=1 for MiNNLO
    
    maxJetParts: maximum number of quarks and gluons (that would result in jets) to include in resulting 'all_particles' array.
                 if less than maxJetParts particles are in an event the rest of the particle vectors have zeroed out attributes
        default=6
    
    allLabel: file name of saved all_particles array
        default='converted_lhe_all_particles'
    
    ttLabel: file name of saved all_particles array
        default='converted_lhe_tt-pair'
    
    recursive: whether to check all subfolders for lhe files
        default=True
    '''
    if recursive==True: # if recursive is turned on, check all subfolders for lhe files, otherwise only check within inputFolder
        lista = glob.glob(inputFolder + "**/*.lhe", recursive=True)  # all .lhe files in inputFolder and all subfolders
    else:
        lista = glob.glob(inputFolder + "*.lhe", recursive=False)  # all .lhe files in inputFolder
    
    # initialize vars
    countEta = 0
    countRapidity = 0

    # initialize arrays
    X0 = []  # all particles array

    num_processes = (mp.cpu_count()-1)  # number of CPU threads to use for the conversion
    pool = mp.Pool(processes=num_processes)
    results = pool.map(process_file_wrapper, zip(lista, [maxJetParts] * len(lista), [theta] * len(lista), [double_jet]*len(lista))) # using multiprocessing, call process_file for each lhe in list, with maxJetParts and theta arguments passed along
    pool.close()
    pool.join()
    
    for lheVector, rapidity, eta in results: # go through the results of calling process_file above and create the arrays to be saved to disk. Also enumartes how many particles or events were skipped due to domain errors
        X0.extend(lheVector)
        countRapidity += rapidity
        countEta += eta

    print("discarded particles: " + str(countRapidity) + ", due to rapidity domain error")
    print("discarded events: " + str(countEta) + ", due to pseudorapidity domain error")
    
    # X0 = np.squeeze(np.array(X0))
    X0 = np.array(X0)

    print("array shape is: "+str(X0.shape)+" and should be: (numEvents, particlesPerEvent + tt-pair, attributesPerParticle)")

    np.savez_compressed(str(outputFolder) + str(outLabel) + str(label)+'.npz', a=X0)
    
    # clear from memory after saving to file
    del countEta
    del countRapidity
    del X0
    del results
    del lheVector


#################################################################################
'''
converted data processing and utilities
'''
#################################################################################


def load_dataset(filePath, i=None): # simply uses np.load to load and return saved datasets 
    with np.load(filePath) as dataset:
        if i is not None:
            return dataset['a'][:, :i, :]
        else:
            return dataset['a']


def trim_datasets(X, Y, shuffle=False):
    '''
    returns inputs X and Y trimed to the length of the shorter of the two
    '''
    if shuffle == True:
        rng = np.random.default_rng()
        rng.shuffle(X)
        rng.shuffle(Y)
    minimum = min(len(X),len(Y))
    X = X[0:minimum,...]
    Y = Y[0:minimum,...]
    return X, Y


def remap_pid(X, pid_i=6):
    
    # remaps the PIDs to small floats -0.6 <=remaped PID<=0.8
    # only looks for quarks and glouns, since that is all we're writing into our arrays
    # returns the input array with remaped PIDs

    # PDGid to small floats dictionary
    PIDmap = {-6: -0.6, 6: 0.6,
              -5: -0.5, 5: 0.5,
              -4: -0.4, 4: 0.4,
              -3: -0.3, 3: 0.3,
              -2: -0.2, 2: 0.2,
              -1: -0.1, 1: 0.1,
              21:  0.8, 0: 0.0}
    
    PIDs = X[:,:,pid_i].reshape((X.shape[0]*X.shape[1]))
    X[:,:,pid_i] = np.asarray([PIDmap[PID] for PID in PIDs]).reshape(X.shape[:2])
    
    return X


def norm(X, ln=False, nrm=None):
    if nrm is not None:
        (mean, std, ln) = nrm
        if ln == True:
            X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
    else:
        if ln == True:
            X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
        mean = np.nanmean(X)
        std = np.nanstd(X)
        nrm = (mean, std, ln)

    X -= mean
    if std >= 1e-2: # mostly for massless gluons not causing divide by zero error
        X /= std
    
    return X, nrm


def normalize_data(X, nrm_array=None):

    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    if nrm_array is None: # calculate normalization
        nrm_array = []
        for particle in range(len(X[0,:,0])):
            nrm_part = []
            for arg in range(6):
                if arg == 0 or arg == 5: # use log for pt and E
                    X[:,particle, arg], nrm_arg = norm(X[:,particle, arg], ln=True) 
                elif arg == 3 and particle == 0: # use log for mass of tt-pair
                    X[:,particle, arg], nrm_arg = norm(X[:,particle, arg], ln=True) 
                else:
                    X[:,particle, arg], nrm_arg = norm(X[:,particle, arg], ln=False)
                nrm_part.append(nrm_arg)
            nrm_array.append(nrm_part)
     
    else: # use given normalization
        for particle, nrm_part in enumerate(nrm_array):
            for arg, nrm_arg in enumerate(nrm_part):
                X[:,particle, arg], _ = norm(X[:,particle, arg], nrm = nrm_arg) 
                    
    # wgt
    X[X[:,:,7] > 0, 7] = 1 #  masks positive weights and sets them = 1
    X[X[:,:,7] < 0, 7] = -1 # masks negative weights and sets them = -1
    
    # PID
    try: X = remap_pid(X)
    except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                           'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    return X, nrm_array


def prep_arrays(X0, X1, val=0.15, shuffle=True, use_class_weights=False):
    '''
    prepare arrays for training
    goes through input arrays X0 and X1 and takes the theta parameter [:,6] and creates the classifier arrays Y0 and Y1 from it
        then removes theta from X0 and X1 arrays
    concatenates X0 and X1 as well as Y0 and Y1. Uses energyflows to_catagorical function on Y to create a one-hot classifier array
    goes through concatenated X and writes the event-weights to weights_array and removes them from X
    then uses energyflows data_split function to create training and validation (15% of all events, by default) arrays from X, Y and weights_array with shuffle (by default)
    returns the data_split arrays X_train, X_val, Y_train, Y_val, wgt_train, wgt_val
    '''
    
    # create weights array from dataset
    class_wgt = 1
    X0_wgt = X0[:,0,-2].copy()
    X1_wgt = X1[:,0,-2].copy()
    if use_class_weights==True:
        class_wgt=len(X0)/len(X1)
        X1_wgt *= class_wgt

    weights_array = np.concatenate((X0_wgt, X1_wgt))
    
    X = np.concatenate((X0[...,:-2], X1[...,:-2]))
    
    # classifier array takes theta parameter from dataset
    Y0 = X0[:,0,-1].copy() # theta is last parameter
    Y1 = X1[:,0,-1].copy()
    Y = np.concatenate((Y0, Y1))
    Y = to_categorical(Y, num_classes=2)
    
    X_train, X_val, Y_train, Y_val, wgt_train, wgt_val = data_split(X, Y, weights_array, train=-1, test=val, shuffle=shuffle)
    
    with tf.device('/cpu:0'):
        X_train = tf.convert_to_tensor(X_train)
        Y_train = tf.convert_to_tensor(Y_train)
        X_val = tf.convert_to_tensor(X_val)
        Y_val = tf.convert_to_tensor(Y_val)
    
    return  X_train, X_val, Y_train, Y_val, np.array(wgt_train), np.array(wgt_val)


def remove_jet_parts(X, maxJetParts = 0):
    '''
    remove (any number of) jet particles from array. By default removes all jet_parts (maxJetParts=0)
    returns input arrays X0 and X1 with all but maxJetParts particles removed from them
    '''
    try: 
        jetParts = len(X[0,:,0]) - 3 # the subtracted 3 particles are the tt-pair, top and anti-top
        for i in range(jetParts - maxJetParts):
            X = np.delete(X, -1, axis=1)
    except: # if this function is called on an array with a different shape, it was probaly done by mistake
        print('Cant remove jet parts, likely wrong shape?')
    return X



#################################################################################
'''
Neural Network 
functions and methods to set up and train the DCTR Neural Network (or load previous training)
Also generates weights for reweighing one dataset into another
'''
#################################################################################



def setup_nn(input_dim=5, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
             loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
             learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', reduceLR = True,
             mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False, verbose = 2):
    
    
    # supported losses
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()

    if loss == 'mse':
        loss = mse_loss
    else:
        loss = cce_loss
    
    # print(f'loss: {loss}')
    # activation functions: if string is unsuppported, fallback to 'relu' or 'softmax'
    supported_acts = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear']
            
    for act in (Phi_acts if isinstance(Phi_acts, (list, tuple)) else [Phi_acts]):
        if act not in supported_acts:
            print(f"unrecognized Phi activation '{act}', falling back to 'relu'")
            act = 'relu'

    for act in (F_acts if isinstance(Phi_acts, (list, tuple)) else [F_acts]):
        if act not in supported_acts:
            print(f"unrecognized F activation '{act}', falling back to 'relu'")
            act = 'relu'
            
    if output_act not in supported_acts:
        print(f"unrecognized output activation '{output_act}', falling back to 'softmax'")
        output_act = 'softmax'
    
    # print('acts set up')
    
    # optimizer
    adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # print('optimizer set up')

    # defines dctr as a particle flow network with given paramterization
    dctr = PFN(input_dim=input_dim, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
               Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, latent_dropout=dropout,
               F_dropouts=dropout, summary=summary, optimizer=adam,
               loss=loss, Phi_acts=Phi_acts, F_acts=F_acts, output_act=output_act) 
    
    # print('set up DCTR')

    # sets up keras checkpoints with monitoring of given metric. monitors 'val_loss' with mode 'min' by default 
    checkpoint = tf.keras.callbacks.ModelCheckpoint(savePath + saveLabel + '.tf',
                                                    monitor = monitor,
                                                    verbose = verbose,
                                                    save_best_only = True,
                                                    mode = mode)
    
    # sets up CSV Logging of callbacks
    # CSVLogger = tf.keras.callbacks.CSVLogger(savePath + saveLabel + '_loss.csv', append=False)
    
    # sets up eraly stopping with given patience (default 15)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = monitor,
                                                     min_delta = 0,
                                                     patience = patience,
                                                     verbose = 1,
                                                     restore_best_weights = True)
    
    # training schedule, reduces learning rate as training commences
    def scheduler(epoch, learning_rate):
        if use_scheduler==False:
            return learning_rate
        elif epoch < 10:
            return learning_rate
        elif epoch > 20:
            return learning_rate * tf.math.exp(-0.02)
        elif epoch > 40:
            return learning_rate * tf.math.exp(-0.03)
        else:
            return learning_rate * tf.math.exp(-0.01)
        
    # scheduler callback
    learn_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    
    # reducesing rate when little improvements are made
    if reduceLR == True:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.6, patience=int(0.4*patience), verbose=1)
    else: 
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=1, patience=int(patience), verbose=0)
    
    callbacks = [checkpoint, EarlyStopping, learn_schedule, reduce_lr] # csv_logger
    
    return dctr, callbacks


def train(dctr, callbacks, X_train, Y_train, X_val, Y_val, wgt_train=1.0, wgt_val=1.0, 
          epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training', verbose = 2, plot=True):
    '''
    method to train the given dctr Neural Network with the X_train/Y_train arrays and validate the predictions with X_val and Y_val
    allows for passing along sample_weights for training and validation. These can be positive and/or negative. If no wgt_train or wgt_val are given, then the weights are set to 1 by default
    plots and saves a figure of loss and accuracy throughout the Epochs
    '''
    #print('starting training')
    history = dctr.fit(X_train, Y_train,
                       sample_weight = pd.Series(wgt_train).to_frame('w_t'), # pd.Series makes the training initialize much, much faster than passing just the weight
                       epochs = epochs,
                       batch_size = batch_size,
                       validation_data = (X_val, Y_val, pd.Series(wgt_val).to_frame('w_v')),
                       verbose = verbose,
                       callbacks = callbacks)
                       
    dctr.save(savePath+saveLabel+'.tf')
    # print(f'saved model: {savePath+saveLabel}.tf')

    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        fig.tight_layout(pad=2)
        
        ax1.plot(history.history['loss'],     label = 'loss', color='cyan')
        ax1.plot(history.history['val_loss'], label = 'val loss', color='blue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('loss')
        ax1.legend()
        
        ax2.plot(history.history['acc'],     label = 'acc', color='pink')
        ax2.plot(history.history['val_acc'], label = 'val acc', color='orange')
        ax2.set_ylabel('acc')
        ax2.set_xlabel('Epochs')
        ax2.legend()
    
        # plt.savefig(savePath+saveLabel+'_history.pdf')
        plt.show()

    min_loss = min(history.history['loss'])

    print('clearing keras session and collecting garbage')
    K.clear_session()
    gc.collect()

    return min_loss

def predict_weights(dctr, X, batch_size=8192, clip=0.00001, verbose=1):
    '''
    generates weights for reweighing X0 to X1: weights_0
                  and for reweighing X1 to X0: weights_1
    from the predictions made by DCTR
    and returns the reweighing arrays
    '''
    predics = dctr.predict(X, batch_size=batch_size, verbose=verbose)
    
    weights = np.divide(predics[:,1], (1-predics[:,1]), out=np.zeros_like(predics[:,1]), where=(predics[:,1])!=1.0 )

    # weights /= np.mean(weights) # adjust weights so that mean is 1

    return weights



def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.build(layer.input_shape)
            
            # Initialize the weights
            weights = [layer.kernel_initializer(layer.kernel.shape),
                       layer.bias_initializer(layer.bias.shape) if layer.bias_initializer else None ]
            
            # Set the weights for the layer
            layer.set_weights(weights)



def get_rwgt(model_list, x0_plt_nrm):
    rwgt_list = []
    for model in model_list:
        K.clear_session()
        with tf.device('/cpu:0'):
            dctr = tf.keras.models.load_model(model)
            rwgt = predict_weights(dctr, x0_plt_nrm[...,:-2], batch_size=8192*8, verbose=0)
        rwgt_list.append(rwgt)
    
    K.clear_session()
    return rwgt_list



def calc_stats(rwgt_list, plt_data, part_indices = [0, 1], arg_indices = [0, 3, 4, 5], stats_only=True, verbose=False):
    print(f'calculating stats for {len(rwgt_list)} models')
    # unpack data
    x0_plt , x0_plt_nrm, x1_plt, x1_plt_wgt = plt_data

    # setup args
    args = [(x1_plt, x1_plt_wgt, '')]
    for rwgt in rwgt_list:
        args.append((x0_plt, rwgt, ''))
    
    mae_all = []
    chi2_all = []
    p_all = []

    # plot with proper ranges set for each observable
    for part_index in part_indices:
        for arg_index in arg_indices:
            div = 31
            if arg_index == 1:  # rapidity
                start = None
                stop = None
            elif arg_index == 3:  # mass
                if part_index == 0:  # tt-pair
                    start = None
                    stop = 1500
                else:
                    start = None
                    stop = None
                    div = 32
            elif arg_index == 4:  # pseudorapidity
                start = -8
                stop = 8
            elif arg_index == 5: # energy
                if part_index == 0:  # tt-pair
                    start = None
                    stop = 3000
                else:
                    start = None
                    stop = 2000
            else:  # pt
                start = 0
                stop = 600
            
            mae_list, chi2_list, p_list = plot_ratio(args, arg_index=arg_index, part_index=part_index, start=start, stop=stop, div=div, stats_only=stats_only, verbose=verbose)
            # print(chi2_list)
            mae_all.append(mae_list)
            chi2_all.append(chi2_list)
            p_all.append(p_list)
            # print(np.array(chi2_all).shape)
            

    # delta phi
    x0_delta_phi = delta_phi(x0_plt)
    x1_delta_phi = delta_phi(x1_plt)
    
    args_delta_phi = [(x1_delta_phi, x1_plt_wgt, '')]
    for rwgt in rwgt_list:
        args_delta_phi.append((x0_delta_phi, rwgt, ''))   
        
    mae_list, chi2_list, p_list = plot_ratio(args_delta_phi, start = 0, stop = math.pi, div = 31, stats_only=stats_only, verbose=verbose)
    
    # print(chi2_list)
    mae_all.append(mae_list)
    chi2_all.append(chi2_list)
    p_all.append(p_list)
    # print(np.array(chi2_all).shape)
    
    # calculate mean over all histograms
    mae_mean_list = np.mean(mae_all, axis=0)
    chi2_mean_list = np.mean(chi2_all, axis=0)
    p_mean_list = np.mean(p_all, axis=0)
    
    # print(f'chi2_list: {chi2_list}')
    # print(f'chi2_mean_list: {chi2_mean_list}')
    
    return mae_mean_list, chi2_mean_list, p_mean_list


def train_run(model, train_data, run, super_epoch, batch_size, epochs, input_dim, Phi_sizes, F_sizes, loss, dropout, l2_reg, Phi_acts, F_acts, output_act, learning_rate, save_dir, label):
    
    # unpack data
    x_train, y_train, x_val, y_val, wgt_train, wgt_val = train_data

    # K.clear_session() # clearing in train() instead
    print(f'starting run {run} of super_epoch {super_epoch} with batch_size {batch_size}')

    # setup nn model
    dctr, callbacks = setup_nn(input_dim=input_dim, Phi_sizes = Phi_sizes, F_sizes = F_sizes, loss = loss, dropout=dropout, l2_reg=l2_reg, Phi_acts=Phi_acts, F_acts=F_acts, output_act=output_act, learning_rate=learning_rate, patience=epochs, savePath=save_dir, saveLabel=label, verbose=0)
    # print('setup neural network')
    if model is None:
        reset_weights(dctr)
        print('reset neural network weights')
    else: # load weights
        dctr = tf.keras.models.load_model(model)
        print(f'loaded neural network model: {model}')
    
    # train using multiprocessing to close child process every repeat to free memory from GPU 
    loss_val = train(dctr, callbacks, x_train, y_train, x_val, y_val, wgt_train=wgt_train, wgt_val=wgt_val, epochs=epochs, batch_size=batch_size, savePath=save_dir, saveLabel=label, verbose=0, plot=False)

    current_model = f'{save_dir}{label}.tf'
    print(f'\n best loss {loss_val:.4f} of run {run} of super_epoch {super_epoch} with batch_size {batch_size}\n')

    return loss_val, current_model


def train_super_epoch(model, train_data, batch_size, repeat, train_dir = '/tf/home/gdrive/_STUDIUM_/DCTR_Paper/train',
                      input_dim=5, Phi_sizes = (100,100,128), F_sizes = (128,100,100), loss = 'mse', dropout=0.0, l2_reg=0.0,
                      Phi_acts=('linear', 'elu', 'gelu'), F_acts=('gelu', 'gelu', 'linear'), output_act='sigmoid', learning_rate=0.001, 
                      epochs = 5, super_epoch = 0):
    
    # training
    model_list = []
    loss_list = []
    for run in range(repeat):

        save_dir = f'{train_dir}/super_epoch_{super_epoch}/run_{run}/'
        label = f's-{super_epoch}_b-{batch_size}_r-{run}'

        loss_val, current_model = train_run(model, train_data, run, super_epoch, batch_size, epochs, input_dim, Phi_sizes, F_sizes, loss, dropout, l2_reg, Phi_acts, F_acts, output_act, learning_rate, save_dir, label)
        
        model_list.append(current_model)
        loss_list.append(loss_val)
        
    return model_list, loss_list
    


def train_super_epoch_choose_best(model, train_data, plt_data, batch_size, repeat, epochs, super_epoch, train_dir = '/tf/home/gdrive/_STUDIUM_/DCTR_Paper/train',
                                  input_dim=5, Phi_sizes = (100,100,128), F_sizes = (128,100,100), loss = 'mse', dropout=0.0, l2_reg=0.0,
                                  Phi_acts=('linear', 'elu', 'gelu'), F_acts=('gelu', 'gelu', 'linear'), output_act='sigmoid', learning_rate=0.001):
    # unpack data
    x0_plt , x0_plt_nrm, x1_plt, x1_plt_wgt = plt_data

    # train and get list of model model
    model_list, loss_list = train_super_epoch(model, train_data, batch_size, repeat, train_dir = train_dir, input_dim=input_dim, Phi_sizes = Phi_sizes, F_sizes = F_sizes, loss = loss, dropout=dropout, l2_reg=l2_reg, Phi_acts=Phi_acts, F_acts=F_acts, output_act=output_act, learning_rate=learning_rate, epochs = epochs, super_epoch = super_epoch)
    
    rwgt_list= get_rwgt(model_list, x0_plt_nrm)
    # stats
    mae_mean_list, chi2_mean_list, p_mean_list = calc_stats(rwgt_list, plt_data)
    min_chi2 = min(chi2_mean_list[1:]) # first is always the baseline x1, always has chi2=0 and is only used for calculating the other statistics
    
    best_where = np.where(chi2_mean_list == min_chi2)
    
    # print(f'best_where: {best_where}')
    # print(f'best_where.shape(): {np.array(best_where).shape}')
    # print(f'best_where[0][0]: {best_where[0][0]}')
    # best_where.shape = (1,1)
    best_model = model_list[best_where[0][0] - 1] # chi2_list which defines best_where includes baseline as first entry, model_list does not
    min_loss = loss_list[best_where[0][0] - 1]

    return best_model, min_chi2, chi2_mean_list, min_loss, loss_list
    

def train_loop(train_data, plt_data, model=None, lowest_chi2 = 1e6, train_dir = './train',
               batch_sizes=[4*8192, 8*8192, 16*8192, 32*8192], repeat=5, super_epochs=35, super_patience = 5, epochs = 8, starting_super_epoch = 1, 
               input_dim=5, Phi_sizes = (100,100,128), F_sizes = (128,100,100), loss = 'mse', dropout=0.0, l2_reg=0.0, 
               Phi_acts=('linear', 'gelu', 'gelu'), F_acts=('gelu', 'gelu', 'linear'), output_act='sigmoid', learning_rate=0.001):
    # device = cuda.get_current_device() # for clearing memory
    # unpack data
    x_train, y_train, x_val, y_val, wgt_train, wgt_val = train_data
    x0_plt , x0_plt_nrm, x1_plt, x1_plt_wgt = plt_data

    patience_counter = 0
    lowest_loss = 1
    best_model_list = []
    lowest_chi2_list = []
    lowest_loss_list = []
    for i in range(super_epochs):
        batch_model_list = []
        batch_chi2_list = []
        batch_loss_list = []
        super_epoch = starting_super_epoch + i
        print(f'starting super_epoch {super_epoch}\n')
        
        # save list of used models for training   
        with open(f'{train_dir}_model_history.csv', 'a', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow([model])
            
        for batch_size in batch_sizes:
            # K.clear_session() # clearing before every run now
            # gc.collect() # collect garbage
            # device.reset()
             
            print(f'starting training with batch_size: {batch_size} and {epochs} epochs\n' +
                  f'starting with weights from model: {model}')
            batch_model, min_chi2, chi2_mean_list, min_loss, loss_list = train_super_epoch_choose_best(model, train_data, plt_data, batch_size, repeat, epochs, super_epoch, train_dir=train_dir, 
                                                                                                       input_dim=input_dim, Phi_sizes = Phi_sizes, F_sizes = F_sizes, loss = loss, 
                                                                                                       Phi_acts=Phi_acts, F_acts=F_acts, output_act=output_act, learning_rate=learning_rate, 
                                                                                                       l2_reg=l2_reg, dropout=dropout)
            
            # save chi2, loss for each run to disk
            for k in range(len(chi2_mean_list)): # one entry for each run, plus baseline (needs to be ignored) x1 as first entry
                if k == 0: continue # first entry in chi2_mean_list is baseline x1
                run = k - 1 
                save_dir = f'{train_dir}/super_epoch_{super_epoch}/run_{run}/'
                label = f'b-{batch_size}_r-{run}'
                # save loss and chi2
                with open(f'{save_dir}{label}_loss_chi2.csv', 'a', newline='\n') as file:
                    writer = csv.writer(file)
                    # len(chi2_mean) = runs + 1 --> use k as index; len(loss_list) = runs --> use run as index
                    writer.writerow([f'super_epoch-{super_epoch}_run-{run}_chi2_and_loss', chi2_mean_list[k], loss_list[run]])

            batch_loss_list.append(min_loss)
            batch_model_list.append(batch_model)
            batch_chi2_list.append(min_chi2)
            
            print(f'\nfinished {repeat} runs of batch_size {batch_size}\n' +
                  f'in super epoch {super_epoch}\n' +
                  f'with best model {batch_model}\n' +
                  f'with chi2 {min_chi2:.4f} and loss {min_loss:.4f}')
            
        # find which batch size had best min_chi2 in completed super_epoch
        best_chi2 = min(batch_chi2_list)
        if best_chi2 < lowest_chi2: # check if there is an improvement from best super epoch
            patience_counter = 0
            lowest_chi2 = best_chi2
            lowest_chi2_list.append(lowest_chi2)
            # set model to best for further training
            best_where = np.where(batch_chi2_list == best_chi2)
            model = batch_model_list[best_where[0][0]]
            best_model_list.append(model)
            lowest_loss = batch_loss_list[best_where[0][0]]
            lowest_loss_list.append(lowest_loss)
        else:
            if patience_counter >= super_patience:
                print('super_patiece reached. Stopping training.')
                break
            elif patience_counter >= math.floor(0.6*super_patience):
                learning_rate = 0.7*learning_rate
            patience_counter += 1
            print(f'no improvement, lowering learnng_rate to {learning_rate}')
            
        print(f'\n\nfinished super_epoch {super_epoch} with {repeat} runs each with batch_sizes:{batch_sizes}\n' +
              f'best model{model}' +
              f'with chi2 {lowest_chi2:.4f} and loss {lowest_loss:.4f}')
    
    best_chi2 = min(batch_chi2_list) # find best after completing all super epochs
    if best_chi2 < lowest_chi2: # check if there is an improvement from best super epoch
        lowest_chi2 = best_chi2
        lowest_chi2_list.append(lowest_chi2)
        # set model to best for further training
        best_where = np.where(batch_chi2_list == best_chi2)
        model = batch_model_list[best_where[0][0]] # no need for - 1 (like in train_super_epoch_choose_best()) b/c chi2 list best_where is based on is build only from models, doesn't include baseline x1 stats
        best_model_list.append(model)
        lowest_loss = batch_loss_list[best_where[0][0]]
        lowest_loss_list.append(lowest_loss)
        
    print('\n\n\n' +
          f'finished loop of {super_epochs} super_epochs\n' +
          f'with batch_sizes:{batch_sizes}\n' +
          f'best model{model}\n' +
          f'with chi2 {lowest_chi2:.4f} and loss {lowest_loss:.4f}')
    
    # save final model to csv of model history
    with open('model_history.csv', 'a', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerow([model])

    return best_model_list, lowest_chi2_list, lowest_loss_list







#################################################################################
'''
plotting functions
used for plotting histograms of datasets 
tt-pair arrays have their own versions of these functions, due to their different shape

functions for plotting 2 or 3 datasets as histograms with options to use weights and custom labels and plotting ranges
Also includes functions for plotting the ratio of X0 and X2 compared to X1, i.e. X0: POWHEG and X2: POWHEG reweighted compared to X1: MiNNLO
'''
#################################################################################


# Global plot settings
# Global plot settings
from matplotlib import rc
import matplotlib.font_manager
import mplhep as hep
plt.style.use(hep.style.CMS)

# rc('text', usetex=True)
# rc('font', size=14)
# rc('xtick', labelsize=10)
# rc('ytick', labelsize=10)
# rc('legend', fontsize=10)

# define dicts of arguments and particles
# [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
# [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]


particles = {0: r't\bar{t}', 
             1: r't',
             2: r'\bar{t}'} 


args_dict = {0: r'p_{T}',
             1: r'y',
             2: r'\phi',
             3: r'm',
             4: r'\eta',
             5: r'E',
             6: r'PID'}       


args_units = {0: r' [GeV]',
              1: r' ',
              2: r' [rad]',
              3: r' [GeV]',
              4: r' ',
              5: r' [GeV]',
              6: r' '}


inverse_units = {0: r' [GeV$^{-1}$]',
                 1: r' ',
                 2: r' [rad$^{-1}$]',
                 3: r' [GeV$^{-1}$]',
                 4: r' ',
                 5: r' [GeV$^{-1}$]',
                 6: r' '}


def plot_weights(wgts, start = -1.5, stop = 2.5, div = 31, title = None):
    bins = np.linspace(start, stop, div)
    plt.figure(figsize=(4,4))
    
    for (wgt, label) in wgts:
        plt.hist(np.clip(wgt, start, stop), bins = bins, label = label, alpha=0.3)
        
    if title is None:
        plt.title('weights')
    else: plt.title(title) 
    
    plt.xlabel(r'weights')
    plt.ylabel(r'counts (log)')
    plt.xlim([start, stop])
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    
def plot_ratio(args, arg_index = 0, part_index = 0, title = None, x_label = None, y_label = None, 
               bins = None, start = None, stop = None, div = 35, ratio_ylim=[0.9,1.1],
               figsize=(6,8), layout='rows', stats_only=False, y_scale=None, verbose = True):
    
    # binning: prio: passed bins, calculated bins from quantiles, linear bins from start, stop, div
    if bins is not None: 
        bins = bins    
    else: # no passed bins, nor optimal bins
        if start is None: # was start/stop given?
            if args[0][0].ndim > 1: # check whether full array
                start = np.min(args[0][0][:,part_index, arg_index])
            else:
                start = np.min(args[0][0])
                
        if stop is None:
            if args[0][0].ndim > 1: # check whether full array
                stop = np.max(args[0][0][:,part_index, arg_index])
            else:
                stop = np.max(args[0][0])
                
        bins = np.linspace(start, stop, div)
    
    start = copy(bins[0])
    stop = copy(bins[-1])
    div = len(bins)
    
    
    if stats_only == False:
        if layout == 'cols':
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
        else: fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
        fig.tight_layout(pad=1.5)
    
    n_list = [] # list of histogram bin counts
    n_sum_list = [] # list of total counts in all bins: used for normalizing
    uncert_list = [] # list of uncertainties sqrt(n) for each bin in each histogram
    mae_list = []
    chi2_list = []
    p_list = []
   #ks_statistic_list = [] 
   #ks_p_value_list = []
    
    for i, (X, wgt, label) in enumerate(args):
        # include events past histogram edges in first/last bin
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # check wheter full dataset is passed or 1D dataset, that can be plotted as is
        if X.ndim > 1:
            n, bins_ = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt)
            bin_indices = np.digitize(X[:,part_index, arg_index], bins = bins)
            # num_evts = len(X[:,0,0])
            # ks_statistic, ks_p_value = stats.ks_2samp(args[0][0][:, part_index, arg_index], X[:, part_index, arg_index])
            
            mean = np.average(X[:,part_index, arg_index], weights=wgt)
            std = math.sqrt(np.absolute(np.average(np.absolute((X[:,part_index, arg_index] - mean))**2, weights=wgt)))
            min, max = np.min(X[:,part_index, arg_index]), np.max(X[:,part_index, arg_index])
            
        else: 
            n, bins_ = np.histogram(X, bins = bins, weights = wgt)
            bin_indices = np.digitize(X, bins = bins)
            # num_evts = len(X)
            # ks_statistic, ks_p_value = stats.ks_2samp(args[0][0], X)
            
            mean = np.average(X, weights=wgt)
            std = math.sqrt(np.average((X - mean)**2, weights=wgt))
            min, max = np.min(X), np.max(X)
        
       #ks_statistic_list.append(ks_statistic)
       #ks_p_value_list.append(ks_p_value)
        
        # statistics
        # uncert: sqrt of the square of all weights in each bin
        uncert = np.array([np.sqrt(np.sum(wgt[bin_indices == bin_index]**2)) for bin_index in range(1, len(bins))])
        uncert_list.append(uncert)
        uncert_nrm = np.divide(uncert, n, out=np.ones_like(uncert), where=(n != 0) )
        uncert_nrm = np.append(uncert_nrm, uncert_nrm[-1]) # extend list by last element for plotting
           
        n_sum = np.nansum(n)
        n_sum_list.append(n_sum)
        # normalize to the expected counts for the first passed X
        n *= (n_sum_list[0] / n_sum)
        n_list.append(n)
        # calculate MAE statistics and chi^2
        mae = np.mean(np.absolute(n_list[0] - n))
        mre = np.mean(np.absolute(np.divide((n_list[0] - n), n, out=np.ones_like(n), where=(n != 0) )))
        
        chi2 = np.nansum(np.power(n_list[0] - n, 2)/(np.power(uncert, 2) + np.power(uncert_list[0], 2)))
        dof = len(bins) - 2 # bins are bin edges not actually bins
        red_chi2 = chi2/dof # reduced chi2
        p = stats.chi2.sf(chi2, dof)
        if verbose: print(f'{label}: mean: {mean:.3f}, std: {std:.3f}, max/min: {max}/{min} \n Mean Absolute Error {mae} \n Mean Relative Error {mre} \n reduced chi square of {red_chi2} with p {p} \n compared to {args[0][2]}')
        mae_list.append(mae)
        chi2_list.append(red_chi2)
        p_list.append(p)
        # plotting
        if stats_only == False:
            if len(args) == 3: # use custom styles for 3 datasets
                line_colors = ['orange', 'blue', 'black']
                line_color = line_colors[i%len(line_colors)]
                line_styles = ['None', 'None', 'dashed']
                line_style = line_styles[i%len(line_styles)]
                ratio_colors = ['orange', 'blue', 'black']
                ratio_color = ratio_colors[i%len(ratio_colors)]
                ratio_styles = ['solid', 'dotted', 'dashed']
                ratio_style = ratio_styles[i%len(ratio_styles)]
                alphas = [0.4, 0.4, 0.0]
                alpha = alphas[i%len(alphas)]
                labels = ['', '', f'{args[i][2]}']
                label_ = labels[i%len(labels)]
                fill_labels = [f'{args[i][2]}', f'{args[i][2]}', '']
                fill_label = fill_labels[i%len(fill_labels)]
            else: # use colors for any datasets
                line_color = f'C{i}'  # Use different colors
                line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
                line_style = line_styles[i%len(line_styles)]
                ratio_style = line_style
                ratio_color = line_color
                alpha=0.2
                fill_label = f'{args[i][2]}'
                label_ = ''
            
            # set bin edges back to start/stop for plotting 
            bins[0] = start
            bins[-1] = stop
            
            # normalize to bin width, so wider bins get shorter. If all bins are same size, nothing happens
            widths = []
            for k in range(len(bins) - 1):
                widths.append(bins[k+1]-bins[k])
            width_div = widths/np.mean(widths) # if all widths are same -> = 1
            # n_tot = np.sum(n)
            hist = np.divide(n, width_div)
            hist = np.append(hist, hist[-1])
            
            ax1.step(bins, hist, label = label_, where = 'post', color=line_color, linestyle=line_style)
            ax1.fill_between(bins, hist, label = fill_label, step='post', alpha=alpha, color=line_color)
            hist = []
            # ratio
            ratio = n/n_list[0] 
            ratio = np.append(ratio, ratio[-1])
            ax2.step(bins, ratio, label = f'{args[i][2]}', where='post', color=ratio_color, linestyle=ratio_style)  # plot ratio compared to first input
            ax2.fill_between(bins, ratio * (1 - uncert_nrm), ratio * (1 + uncert_nrm), alpha=0.3, step='post', color=ratio_color) 
            ratio = []
        
    if stats_only == False:
        if title is None:
            ax1.set_title(str(particles.get(part_index, 'Jet'))+ ': ' + str(args_dict.get(arg_index, 'Jet')))
        else: ax1.set_title(title)
        
        if x_label is None:
            ax1.set_xlabel(str(str(particles.get(part_index, 'Jet'))+': '+args_dict.get(arg_index, 'Jet')))
        else: ax1.set_xlabel(x_label)
            
        if y_label is None:
            ax1.set_ylabel(f'expected count \n normalized to {args[0][2]}')
        else: ax1.set_ylabel(y_label)
        ax1.set_xlim([start, stop])
        if y_scale == 'log':
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=1e-12)
        else:
            ax1.set_ylim(bottom=0)
        ax1.legend()
        
        ax2.set_title('ratio plot')
        ax2.set_xlim([start, stop])
        ax2.set_ylim(ratio_ylim)
        ax2.set_ylabel('ratio')
        ax2.legend()
        
        plt.show()
    
    return mae_list, chi2_list, p_list # ks_statistic_list, ks_p_value_list



pythia_text = r'$POWHEG \; pp \to  t\bar{t}$'
def make_legend(ax, title):
    leg = ax.legend(frameon=False)
    leg.set_title(title, prop={'size':20})
    leg.texts[0].set_fontsize(20)
    leg._legend_box.align = "left"
    plt.tight_layout()



def plot_ratio_cms(args, arg_index = 0, part_index = 0, title = None, x_label = None, y_label = None, bins = None, start = None, stop = None, div = 35, ratio_ylim=[0.9,1.1], density=True, pythia_text = pythia_text, figsize=(8,10), y_scale=None, overflow=False, hep_text = 'Simulation Preliminary', center_mass_energy = '13 TeV', part_label=None, arg_label=None, unit=None, inv_unit=None):

    try:
        [(x1, x1_wgt , x1_label), (x0, x0_wgt , x0_label), (x0, x0_rwgt, x0_rwgt_label)] = args
    except:
        print('args not in right form. Needs to be args = [(x1, x1_wgt, x1_label), (x0, x0_wgt, x0_label), (x0, x0_rwgt, x0_rwgt_label)]')
    
    plt_style_10a = {'color':'Green', 'linewidth':3, 'linestyle':'--'} #, 'density':True, 'histtype':'step'}
    plt_style_11a = {'color':'black', 'linewidth':3, 'linestyle':'-'} #', 'density':True, 'histtype':'step'}
    plt_style_12a = {'color':'#FC5A50', 'linewidth':3, 'linestyle':':'} #, 'density':True, 'histtype':'step'}

    # binning: prio: passed bins, calculated bins from quantiles, linear bins from start, stop, div
    if bins is not None: 
        bins = bins    
    else: # no passed bins, nor optimal bins
        if start is None: # was start/stop given?
            if args[0][0].ndim > 1: # check whether full array
                start = np.min(args[0][0][:,part_index, arg_index])
            else:
                start = np.min(args[0][0])
                
        if stop is None:
            if args[0][0].ndim > 1: # check whether full array
                stop = np.max(args[0][0][:,part_index, arg_index])
            else:
                stop = np.max(args[0][0])
                
        bins = np.linspace(start, stop, div)
    
    start = copy(bins[0])
    stop = copy(bins[-1])
    div = len(bins)

    if overflow is True:
        # include events past histogram edges in first/last bin
        bins[0] = -np.inf
        bins[-1] = np.inf
    
    n_list = [] # list of histogram bin counts
    n_sum_list = [] # list of total counts in all bins: used for normalizing
    uncert_list = [] # list of uncertainties for each bin in each histogram
    uncert_nrm_list = []
    dense_list = []

    for i, (X, wgt, label) in enumerate(args):
        # check wheter full dataset is passed or 1D dataset, that can be plotted as is
        if X.ndim > 1:
            n, bin_edges = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt) # calculate histogram
            if density is True:
                dense_n, bin_edges = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt, density=True)
                dense_n = np.append(dense_n, dense_n[-1])
                dense_list.append(dense_n) # extend list by last element for plotting
            bin_indices = np.digitize(X[:,part_index, arg_index], bins = bins) # which bin did each event end up in
            
        else: 
            n, bin_edges = np.histogram(X, bins = bins, weights = wgt)
            if density is True:
                dense_n, bin_edges = np.histogram(X, bins = bins, weights = wgt, density=True)
                dense_n = np.append(dense_n, dense_n[-1])
                dense_list.append(dense_n) # extend list by last element for plotting
            bin_indices = np.digitize(X, bins = bins) # which bin did each event end up in
        

        # statistics
        # uncert: sqrt of the square of all weights in each bin
        uncert = np.array([np.sqrt(np.sum(wgt[bin_indices == bin_index]**2)) for bin_index in range(1, len(bins))])
        uncert_list.append(uncert)
        uncert_nrm = np.divide(uncert, n, out=np.ones_like(uncert), where=(n != 0)) # normalizing uncert by dividing by bin counts, for non-zero bin counts

        uncert_nrm = np.append(uncert_nrm, uncert_nrm[-1]) # extend list by last element for plotting
        uncert_nrm_list.append(uncert_nrm)
        
        n_sum = np.nansum(n) # total number of events in all bins
        n_sum_list.append(n_sum)
        # normalize to the expected counts for the first passed X
        n *= (n_sum_list[0] / n_sum)
        n = np.append(n, n[-1]) # extend array by repeating last element for plotting
        n_list.append(n)
    
    
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout(pad=1)

    if density is True:
        hist0 = dense_list[0] # NNLO
        hist1 = dense_list[1] # NLO
        hist2 = dense_list[2] # NLO rwgt
    else:
        hist0 = n_list[0] # NNLO
        hist1 = n_list[1] # NLO
        hist2 = n_list[2] # NLO rwgt

    # First subplot
    bin_edges[0] = start # reset bin edges for plotting
    bin_edges[-1] = stop
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    axes[0].step(bin_edges, hist0, label = x1_label, where='post', **plt_style_11a)
    axes[0].step(bin_edges, hist1, label = x0_label, where='post', **plt_style_10a)
    axes[0].step(bin_edges, hist2, label = x0_rwgt_label, where='post', **plt_style_12a)

    # Calculate the ratios of histograms
    ratio_0 = hist0 / hist0
    ratio_1 = hist1 / hist0
    ratio_2 = hist2 / hist0


    # labels and titles
    make_legend(axes[0], pythia_text)

    if part_label is None:
        part = particles.get(part_index)
    else:
        part = part_label

    if arg_label is None:
        obs = args_dict.get(arg_index)
    else:
        obs = arg_label
    
    if unit is None:
        inv_unit = inverse_units.get(arg_index)
        unit = args_units.get(arg_index)
    else:
        inv_unit = inv_unit
        unit = unit
    
    # Constructing the label using Python string formatting
    label = r'$1$/$\sigma \frac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)

    axes[0].set_ylabel(label)

    if y_scale == 'log':
        axes[0].set_yscale('log')
    else: 
        axes[0].set_ylim(bottom=0)
    axes[0].grid(True)

    # Second subplot
    axes[1].errorbar(bin_centers, ratio_0[:-1], yerr=uncert_nrm_list[0][:-1], fmt='-', color='black')
    axes[1].errorbar(bin_centers, ratio_1[:-1],   yerr=uncert_nrm_list[1][:-1], fmt='--', color='green')
    axes[1].errorbar(bin_centers, ratio_2[:-1],    yerr=uncert_nrm_list[2][:-1], fmt=':', color='#FC5A50')
    axes[1].plot([start, stop], [1,1], '-', color='black',  linewidth=3, label=x1_label)
    axes[1].plot(bin_centers, ratio_1[:-1],  '--', color='green',  linewidth=3, label=x0_label)
    axes[1].plot(bin_centers, ratio_2[:-1],    ':', color='#FC5A50',linewidth=3, label=x0_rwgt_label)
    axes[1].set_xlabel(fr'${obs}({part}){unit}$')
    axes[1].set_ylabel(f'Ratio(/NNLO)')
    axes[1].grid(True)

    # print(f'uncertainty NLO: {uncert_nrm_list[0]}')
    
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
    axes[1].set_ylim(ratio_ylim)

    axes[0].set_xlim([start,stop])
    axes[1].set_xlim([start,stop])
    axes[1].legend(fontsize=13)

    #hep.cms.label(ax=axes[0], data=False, paper=False, lumi=None, fontsize=20, loc=0)
    hep.cms.text(hep_text, loc=0, fontsize=20, ax=axes[0])
    axes[0].text(1.0, 1.05, center_mass_energy, ha="right", va="top", fontsize=20, transform=axes[0].transAxes)

    # Save the figure
    plt.savefig(f'./plots/{obs}_{part}_plot.pdf')
    plt.show()



def plot_ratio_cms_4(args, arg_index = 0, part_index = 0, title = None, x_label = None, y_label = None, bins = None, start = None, stop = None, div = 35, ratio_ylim=[0.9,1.1], density=True, pythia_text = pythia_text, figsize=(8,10), y_scale=None, binning = 'linear', overflow=False, hep_text = 'Simulation Preliminary', center_mass_energy = '13 TeV', part_label=None, arg_label=None, unit=None, inv_unit=None):

    try:
        [(x1, x1_wgt , x1_label), (x0, x0_wgt , x0_label), (x0, x0_rwgt, x0_rwgt_label), (x0, x0_rwgt_alt, x0_rwgt_alt_label)] = args
    except:
        print('args not in right form. Needs to be args = [(x1, x1_wgt, x1_label), (x0, x0_wgt, x0_label), (x0, x0_rwgt, x0_rwgt_label), (x0, x0_rwgt_alt, x0_rwgt_alt_label)]')
    
    plt_style_10a = {'color':'Green', 'linewidth':3, 'linestyle':'--'} #, 'density':True, 'histtype':'step'}
    plt_style_11a = {'color':'black', 'linewidth':3, 'linestyle':'-'} #', 'density':True, 'histtype':'step'}
    plt_style_12a = {'color':'#FC5A50', 'linewidth':3, 'linestyle':':'} #, 'density':True, 'histtype':'step'}
    plt_style_13a = {'color':'blue', 'linewidth':3, 'linestyle':'-.'} #, 'density':True, 'histtype':'step'}


    # binning: prio: passed bins, calculated bins from quantiles, linear bins from start, stop, div
    if bins is not None: 
        bins = bins    
    else: # no passed bins, nor optimal bins
        if start is None: # was start/stop given?
            if args[0][0].ndim > 1: # check whether full array
                start = np.min(args[0][0][:,part_index, arg_index])
            else:
                start = np.min(args[0][0])
                
        if stop is None:
            if args[0][0].ndim > 1: # check whether full array
                stop = np.max(args[0][0][:,part_index, arg_index])
            else:
                stop = np.max(args[0][0])
        if binning != 'log':        
            bins = np.linspace(start, stop, div)
        else:
            if start == 0:
                start = 1
            bins = np.logspace(np.log10(start), np.log10(stop), div)
     
    start = copy(bins[0])
    stop = copy(bins[-1])
    div = len(bins)

    if overflow is True:
        # include events past histogram edges in first/last bin
        bins[0] = -np.inf
        bins[-1] = np.inf
    
    n_list = [] # list of histogram bin counts
    n_sum_list = [] # list of total counts in all bins: used for normalizing
    uncert_list = [] # list of uncertainties for each bin in each histogram
    uncert_nrm_list = []
    dense_list = []

    for i, (X, wgt, label) in enumerate(args):
        # check wheter full dataset is passed or 1D dataset, that can be plotted as is
        if X.ndim > 1:
            n, bin_edges = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt) # calculate histogram
            if density is True:
                dense_n, bin_edges = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt, density=True)
                dense_n = np.append(dense_n, dense_n[-1])
                dense_list.append(dense_n) # extend list by last element for plotting
            bin_indices = np.digitize(X[:,part_index, arg_index], bins = bins) # which bin did each event end up in
            
        else: 
            n, bin_edges = np.histogram(X, bins = bins, weights = wgt)
            if density is True:
                dense_n, bin_edges = np.histogram(X, bins = bins, weights = wgt, density=True)
                dense_n = np.append(dense_n, dense_n[-1])
                dense_list.append(dense_n) # extend list by last element for plotting
            bin_indices = np.digitize(X, bins = bins) # which bin did each event end up in
        

        # statistics
        # uncert: sqrt of the square of all weights in each bin
        uncert = np.array([np.sqrt(np.sum(wgt[bin_indices == bin_index]**2)) for bin_index in range(1, len(bins))])
        uncert_list.append(uncert)
        uncert_nrm = np.divide(uncert, n, out=np.ones_like(uncert), where=(n != 0)) # normalizing uncert by dividing by bin counts, for non-zero bin counts

        uncert_nrm = np.append(uncert_nrm, uncert_nrm[-1]) # extend list by last element for plotting
        uncert_nrm_list.append(uncert_nrm)
        
        n_sum = np.nansum(n) # total number of events in all bins
        n_sum_list.append(n_sum)
        # normalize to the expected counts for the first passed X
        n *= (n_sum_list[0] / n_sum)
        n = np.append(n, n[-1]) # extend array by repeating last element for plotting
        n_list.append(n)
    
    
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout(pad=1)

    if density is True:
        hist0 = dense_list[0] # NNLO
        hist1 = dense_list[1] # NLO
        hist2 = dense_list[2] # NLO rwgt
        hist3 = dense_list[3] # bins rwgt
    else:
        hist0 = n_list[0] # NNLO
        hist1 = n_list[1] # NLO
        hist2 = n_list[2] # NLO rwgt
        hist3 = n_list[3] # bins rwgt

    # First subplot
    bin_edges[0] = start # reset bin edges for plotting
    bin_edges[-1] = stop
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    axes[0].step(bin_edges, hist0, label = x1_label, where='post', **plt_style_11a)
    axes[0].step(bin_edges, hist1, label = x0_label, where='post', **plt_style_10a)
    axes[0].step(bin_edges, hist2, label = x0_rwgt_label, where='post', **plt_style_12a)
    axes[0].step(bin_edges, hist3, label = x0_rwgt_alt_label, where='post', **plt_style_13a)

    # Calculate the ratios of histograms
    ratio_0 = hist0 / hist0
    ratio_1 = hist1 / hist0
    ratio_2 = hist2 / hist0
    ratio_3 = hist3 / hist0

    # labels and titles
    make_legend(axes[0], pythia_text)

    if part_label is None:
        part = particles.get(part_index)
    else:
        part = part_label

    if arg_label is None:
        obs = args_dict.get(arg_index)
    else:
        obs = arg_label
    
    if unit is None:
        inv_unit = inverse_units.get(arg_index)
        unit = args_units.get(arg_index)
    else:
        inv_unit = inv_unit
        unit = unit
    
    # Constructing the label using Python string formatting
    label = r'$1$/$\sigma \frac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)

    axes[0].set_ylabel(label)

    if y_scale == 'log':
        axes[0].set_yscale('log')
    else: 
        axes[0].set_ylim(bottom=0)
    axes[0].grid(True)

    # Second subplot
    axes[1].errorbar(bin_centers, ratio_0[:-1], yerr=uncert_nrm_list[0][:-1], fmt='-', color='black')
    axes[1].errorbar(bin_centers, ratio_1[:-1],   yerr=uncert_nrm_list[1][:-1], fmt='--', color='green')
    axes[1].errorbar(bin_centers, ratio_2[:-1],    yerr=uncert_nrm_list[2][:-1], fmt=':', color='#FC5A50')
    axes[1].errorbar(bin_centers, ratio_3[:-1],    yerr=uncert_nrm_list[3][:-1], fmt='-.', color='blue')
    axes[1].plot([start, stop], [1,1], '-', color='black',  linewidth=3, label=x1_label)
    axes[1].plot(bin_centers, ratio_1[:-1],  '--', color='green',  linewidth=3, label=x0_label)
    axes[1].plot(bin_centers, ratio_2[:-1],    ':', color='#FC5A50',linewidth=3, label=x0_rwgt_label)
    axes[1].plot(bin_centers, ratio_3[:-1],    ':', color='blue', linewidth=3, label=x0_rwgt_alt_label)
    axes[1].set_xlabel(fr'${obs}({part}){unit}$')
    axes[1].set_ylabel(f'Ratio(/NNLO)')
    axes[1].grid(True)

    # print(f'uncertainty NLO: {uncert_nrm_list[0]}')
    
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
    axes[1].set_ylim(ratio_ylim)

    axes[0].set_xlim([start,stop])
    axes[1].set_xlim([start,stop])
    axes[1].legend(fontsize=13)

    #hep.cms.label(ax=axes[0], data=False, paper=False, lumi=None, fontsize=20, loc=0)
    hep.cms.text(hep_text, loc=0, fontsize=20, ax=axes[0])
    axes[0].text(1.0, 1.05, center_mass_energy, ha="right", va="top", fontsize=20, transform=axes[0].transAxes)

    # Save the figure
    plt.savefig(f'./plots/{obs}_{part}_plot.pdf')
    plt.show()