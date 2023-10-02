#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: finn puschmann
"""


# energyflow dependencies import
from __future__ import absolute_import, division, print_function
import tensorflow.keras.backend as K
import tensorflow as tf

# system modules
import sys
import os
import glob
import math 
import multiprocessing as mp

# standard numerical library imports
import numpy as np
from scipy import stats
from math import sqrt, atan2, log
import matplotlib.pyplot as plt
from hist import intervals
import mplhep
import pandas as pd

# energyflow imports
from energyflow.archs import PFN
from energyflow.utils import data_split, to_categorical

# madgraph imports
sys.path.append('/tf/madgraph/MG5_aMC_v2_9_16')
try:
    from madgraph.various.lhe_parser import FourMomentum, EventFile
except ModuleNotFoundError:
    print('Madgraph was not found in PATH or in docker /tf/madgraph/MG5_aMC_v2_9_16 dir \n you can added temporarily with sys.path.append(\'path/to/madgraph\')')

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


def process_file(filename, maxJetParts, theta):
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
    lheVector = []
    countEta = 0
    countRapidity = 0

    for event in lhe1: # goes through the lhe file, event by event
        try:
            eventVector, rapidity = process_event(event, maxJetParts, theta) # calls the process_event function for every event in lhe file. 
        # if there is a problem with process_event it means the tt-pair pseudorapidity could not be calculated, countEta is updated to keep track of how many events were skipped because of this.
        except ValueError: 
            countEta += 1
            continue
        except ZeroDivisionError:
            countEta += 1
            continue
        
        # append current event vectors to the rest of the event vectors of the lhe file
        lheVector.append(eventVector)
        countRapidity += rapidity

    return lheVector, countRapidity, countEta


def pseudorapidity(particle):
    pz = particle.pz
    p = np.sqrt(np.power(particle.px, 2) + np.power(particle.py, 2) + np.power(particle.pz, 2))
    pseudorapidity = 0.5*math.log((p-pz)/(p+pz)) # if this has a domain error, it is caught by process_file() and a counter for failed pseudorapidities is kept.
    return pseudorapidity


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

def process_event(event, maxJetParts, theta):
    '''
    is called by process_file for every event in it's LHE file
    inherits filename, maxJetParts and theta from process_file and thus from convertLHE
    
    goes through an event particle by particle, adding the top, anti-top,
    as well as upto maxJetParts other quarks and glouns in an event to arrays.
    Callculates some properties for the particles, like phi and pseudorapidity and discards
    some particles and events if it comes to domainErrors.
    
    returns the eventVector with all particles including tt-pair of the event
    '''
    w = event.wgt
    ptop = FourMomentum(0, 0, 0, 0)
    pantitop = FourMomentum(0, 0, 0, 0)
    pjet = FourMomentum(0, 0, 0, 0)
    eventVector = []
    eventJetVector = []
    countRapidity = 0
    
    # particle processing
    for particle in event: # loops through every particle in event
        if (particle.status==2): # particle.status = 2: unstable particles, here only the top or anti-top are saved, W Bosons are ignored
            if particle.pid==6: # top quark
                if check_rapidity(particle)==True: # if rapidity is fine
                    ptop = FourMomentum(particle) # create FourMomentum for top
                else:
                    countRapidity += 1
                    continue
                
            elif particle.pid==-6: # anti-top quark
                if check_rapidity(particle)==True: # if rapidity is fine
                    pantitop = FourMomentum(particle) # create FourMomentum for anti-top
                else:
                    countRapidity += 1
                    continue
                
            else: continue
        
        if (particle.status==1): # particle.status = 1: stable particles, here only jet particles (quarks and gluons) considered
            if ((particle.pid<6 and particle.pid>-6) or particle.pid==21): # only quarks and glouns are saved
                if len(eventJetVector) < maxJetParts: #limit to maxJetParts particles per event, to avoid ragged arrays, if the event has less particles than the rest is filled with zeros
                    if check_rapidity(particle)==True: # if rapidity is fine
                        pjet=FourMomentum(particle) # FourMomentum of particle in Jet
                        eventJetVector.append([pjet.pt, pjet.rapidity, phi(pjet), pjet.mass, pseudorapidity(pjet), pjet.E, particle.pid, w, theta]) # add particle to Jet Vector of event
                    else:
                        countRapidity += 1
                        continue
                    
                else: continue

            else: continue
        
        else: continue
    
    # Top pair processing
    p_tt = ptop + pantitop # create madgraph FourMomentum of tt-pair
    
    # pseudorapidity
    tt_pseudorapidity = pseudorapidity(p_tt)
    top_pseudorapidity = pseudorapidity(ptop)
    anti_pseudorapidity = pseudorapidity(pantitop)

    # for each event: 1. tt-pair, 2. top, 3. anti-top, followed by maxJetParts of jet particles
    eventVector.append([p_tt.pt,     p_tt.rapidity,     phi(p_tt),     p_tt.mass,     tt_pseudorapidity,   p_tt.E,      0, w, theta]) # add tt-pair to output array
    eventVector.append([ptop.pt,     ptop.rapidity,     phi(ptop),     ptop.mass,     top_pseudorapidity,  ptop.E,      6, w, theta]) # add top quark to event vector
    eventVector.append([pantitop.pt, pantitop.rapidity, phi(pantitop), pantitop.mass, anti_pseudorapidity, pantitop.E, -6, w, theta]) # add anti-top quark to event vector
    
    if len(eventJetVector) == maxJetParts: # check length of eventJetVector to avoid ragged arrays
        eventVector.extend(eventJetVector) # extend the event vector with the maxJetParts length jet vector
    else: 
        for i in range(maxJetParts):
            if len(eventJetVector) < maxJetParts:
                eventJetVector.append([0, 0, 0, 0, 0, 0, 0, 0, 0]) # pad eventJetVector to length of max particles to avoid ragged arrays
            else: continue
        eventVector.extend(eventJetVector) # extend the event vector with the maxJetParts length jet vector

    return eventVector, countRapidity


def process_file_wrapper(args):
    '''
    wrapper for multiprocessing inside convert lhe
    simply calls the process_file function
    '''
    filename, maxJetParts, theta = args
    return process_file(filename, maxJetParts, theta)


def convert_lhe(inputFolder, outputFolder, theta, outLabel=None, maxJetParts=8, label='converted_lhe', recursive=True):
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
    results = pool.map(process_file_wrapper, zip(lista, [maxJetParts] * len(lista), [theta] * len (lista))) # using multiprocessing, call process_file for each lhe in list, with maxJetParts and theta arguments passed along
    pool.close()
    pool.join()

    for lheVector, rapidity, eta in results: # go through the results of calling process_file above and create the arrays to be saved to disk. Also enumartes how many particles or events were skipped due to domain errors
        X0.extend(lheVector)
        countRapidity += rapidity
        countEta += eta

    print("discarded particles: " + str(countRapidity) + ", due to rapidity domain error")
    print("discarded events: " + str(countEta) + ", due to tt-pair pseudorapidity domain error")
    
    # X0 = np.squeeze(np.array(X0))
    X0 = np.array(X0)

    print("array shape is: "+str(X0.shape)+" and should be: (numEvents, particlesPerEvent + tt-pair, attributesPerParticle)")

    np.savez_compressed(str(outputFolder) + str(outLabel) + str(label)+'.npz', a=X0)
    
    # clear from memory after saving to file
    countEta = 0
    countRapidity = 0
    X0 = []
    results = []
    lheVector = []

#################################################################################
'''
converted data processing and utilities
'''
#################################################################################


def load_dataset(filePath): # simply uses np.load to load and return saved datasets 
    dataset=np.load(filePath)
    X=dataset['a']
    return X


def trim_datasets(X, Y):
    '''
    returns inputs X and Y trimed to the length of the shorter of the two
    '''
    minimum = min(len(X[:,0]),len(Y[:,0]))
    X = X[0:minimum,:]
    Y = Y[0:minimum,:]
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


def norm(X, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
    mean = np.mean(X)
    std = np.std(X)
    norm = (mean, std, ln)
    X -= mean
    X /= std
    return X, norm

def norm_2(X, Y, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
        Y = np.log(np.clip(Y, a_min = 1e-6, a_max = None))
    mean = np.mean(np.concatenate((X, Y)))
    std = np.std(np.concatenate((X, Y)))
    norm = (mean, std, ln)
    X -= mean
    X /= std
    
    Y -= mean
    Y /= std
    return X, Y, norm

def un_norm(X, norm):
    mean, std, ln = norm
    X *= std
    X += mean
    if ln == True:
        X = np.exp(X)
    return X

def un_norm_2(X, Y, norm):
    mean, std, ln = norm
    X *= std
    X += mean
    if ln == True:
        X = np.exp(X)
    Y *= std
    Y += mean
    if ln == True:
        Y = np.exp(Y)
    return X, Y


def norm_wgt(X):
    mode = stats.mode(np.absolute(X), keepdims=False)[0]
    X /= mode
    return X


def normalize_data(X, pt=True, rapidity=True,  phi=True, mass=True, 
                   PID=True, wgt=True, pseudorapidity=True, energy=True):
    
    norm_dict = {}
    norm_pt = (0.0, 1.0, True)
    norm_rapidity = (0.0, 1.0, False)
    norm_phi = (0.0, 1.0, False)
    norm_mass = (0.0, 1.0, True)
    norm_pseudorapidity = (0.0, 1.0, False)
    norm_e = (0.0, 1.0, False)
    
    num_parts = len(X[0,:,0])
    
    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    if pt==True: # 0
        X[...,0], norm_pt = norm(X[...,0], ln=True)
    
    if rapidity==True: # 1
        r_std = np.std(X[:,1,1])
        X[...,1] = np.divide(X[...,1], r_std)
        norm_rapidity = (0.0, r_std, False)
        
    if phi==True: # 2
        X[...,2] = X[...,2]/math.pi
        norm_phi = (0.0, math.pi, False)
    
    if mass==True: # 3
        X[...,3], norm_mass = norm(X[...,3], ln=True)
    
    if pseudorapidity==True: # 4
        pr_std = np.std(X[:,1,4])
        X[...,4] = np.divide(X[...,1], pr_std)
        norm_pseudorapidity  = (0.0, pr_std, False)
    
    if energy==True: # 5
        X[...,5], norm_e = norm(X[...,5], ln=True)
    
    if PID==True: # 6
        try: X = remap_pid(X)
        except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                               'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    
    if wgt==True: # 7
            X[...,7] = norm_wgt(X[:,0,7]) * num_parts
    
    norm_dict = {'pt':             norm_pt,
                 'rapidity':       norm_rapidity,
                 'phi':            norm_phi,
                 'mass':           norm_mass,
                 'pseudorapidity': norm_pseudorapidity,
                 'energy':         norm_e}
    
    return X, norm_dict


def normalize_data_2(X, Y, pt=True, rapidity=True,  phi=True, mass=True, 
                   PID=True, wgt=True, pseudorapidity=True, energy=True):
    
    norm_dict = {}
    norm_pt = (0.0, 1.0, False)
    norm_rapidity = (0.0, 1.0, False)
    norm_phi = (0.0, 1.0, False)
    norm_mass = (0.0, 1.0, False)
    norm_pseudorapidity = (0.0, 1.0, False)
    norm_e = (0.0, 1.0, False)
    
    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    num_parts_X = len(X[0,:,0])
    num_parts_Y = len(Y[0,:,0])
    
    if pt==True: # 0
        X[...,0], Y[...,0], norm_pt = norm_2(X[...,0], Y[...,0], ln=True)
    
    if rapidity==True: # 1
        r_std = np.std(np.concatenate((X[:,1,1], Y[:,1,1])) )
        X[...,1] = np.divide(X[...,1], r_std)
        Y[...,1] = np.divide(Y[...,1], r_std)
        norm_rapidity = (0.0, r_std, False)
        
    if phi==True: # 2
        X[...,2] = np.divide(X[...,2], math.pi)
        Y[...,2] = np.divide(Y[...,2], math.pi)
        norm_phi = (0.0, math.pi, False)
    
    if mass==True: # 3
        X[...,3], Y[...,3], norm_mass = norm_2(X[...,3], Y[...,3], ln=True)
    
    if pseudorapidity==True: # 4
        pr_std = np.std(np.concatenate((X[:,1,4], Y[:,1,4])) )
        X[...,4] = np.divide(X[...,1], pr_std)
        Y[...,4] = np.divide(Y[...,1], pr_std)
        norm_pseudorapidity  = (0.0, pr_std, False)
    
    if energy==True: # 5
        X[...,5], Y[...,5], norm_e = norm_2(X[...,5], Y[...,5], ln=True)
    
    if PID==True: # 6
        try: X = remap_pid(X)
        except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                               'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
        try: Y = remap_pid(Y)
        except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                               'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    
    if wgt==True: # 7
            X[...,7] = np.transpose([norm_wgt(X[:,0,7])] * num_parts_X)
            Y[...,7] = np.transpose([norm_wgt(Y[:,0,7])] * num_parts_Y)
    
    norm_dict = {'pt':             norm_pt,
                 'rapidity':       norm_rapidity,
                 'phi':            norm_phi,
                 'mass':           norm_mass,
                 'pseudorapidity': norm_pseudorapidity,
                 'energy':         norm_e}
    
    return X, Y, norm_dict


def un_normalize_data(X, norm_dict):
    for i, norm in enumerate(norm_dict.values()):
        X[...,i] = un_norm(X[...,i], norm)
            
    return X 

def un_normalize_data_2(X, Y, norm_dict):
    for i, norm in enumerate(norm_dict.values()):
        X[...,i], X[...,i] = un_norm_2(X[...,i], X[...,i], norm)
            
    return X , Y


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
    Y0 = []
    Y1 = []
    # classifier array takes theta parameter from dataset
    Y0 = X0[:,0,8] 
    Y1 = X1[:,0,8]
    
    # removing theta as it was already used to create the classifier arrays Y0 and Y1
    X0 = np.delete(X0, -1, -1) # theta paramater is last argument in last axis
    X1 = np.delete(X1, -1, -1)
    
    X = []
    X = np.concatenate((X0, X1))
    
    Y = []
    Y = np.concatenate((Y0, Y1))
    Y = to_categorical(Y, num_classes=2)
    
    class_wgt = 1
    if use_class_weights==True:
        class_wgt=len(X0)/len(X1)
        X0[...,7] /= class_wgt
    weights_array = X[:,0,7] 
    
    # removing weights as it was already used to create the weights_array
    X = np.delete(X, -1, -1) # event weight paramater is last argument (after theta is removed) in last axis for both all particles and tt-pair arrays
    
    X_train, X_val, Y_train, Y_val, wgt_train, wgt_val = data_split(X, Y, weights_array, train=-1, test=val, shuffle=shuffle)
    
    return  X_train, X_val, Y_train, Y_val, np.array(wgt_train), np.array(wgt_val)


def remove_jet_parts(X0, X1, maxJetParts = 0):
    '''
    remove (any number of) jet particles from array. By default removes all jet_parts (maxJetParts=0)
    returns input arrays X0 and X1 with all but maxJetParts particles removed from them
    '''
    try: 
        jetParts = len(X0[0,:,0]) - 3 # the subtracted 3 particles are the tt-pair, top and anti-top
        for i in range(jetParts - maxJetParts):
            X0 = np.delete(X0, -1, axis=1)
            X1 = np.delete(X1, -1, axis=1)
    except: # if this function is called on an array with a different shape, it was probaly done by mistake
        print('Cant remove jet parts, likely wrong shape?')
    return X0, X1



#################################################################################
'''
Neural Network 
functions and methods to set up and train the DCTR Neural Network (or load previous training)
Also generates weights for reweighing one dataset into another
'''
#################################################################################



def setup_nn(input_dim=7, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
             use_custom_loss=False, dropout=0.0, l2_reg=0.0, learning_rate=0.001,
             patience=10, use_scheduler=True, use_focal=False, gamma=1.1,
             monitor='val_loss', mode='min', savePath=currentPath,
             saveLabel='DCTR_training', summary=False):
    '''
    l2_reg range that worked well: (1e-7, 5e-6)
    Uses energyflows Particle Flow Network (PFN) architecture to setup the DCTR neural network. 
    '''
    
    focal_loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    
    def custom_loss(y_true, y_pred):
        pred = K.clip(y_pred, 0.000001, 0.9999999)
        rwgt = tf.divide(pred, tf.add(1.0, -pred))
        rwgt_penalty = K.mean(K.pow(K.log(rwgt), 10))
        
        my_loss =  (1 + 0.001*rwgt_penalty + 0.1*focal_loss(y_true, y_pred)) * cce_loss(y_true, y_pred) # (1 + focal_loss(y_true, y_pred, sample_weights)) *
        
        return K.mean(my_loss)
    
    loss = custom_loss
    
    # if use_custom_loss == True:
        # loss = custom_loss
    # elif use_focal == True:
        # loss = focal_loss
    # else:
        # loss = cce_loss
    
    
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
    
    adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # defines dctr as a particle flow network with given input_dim, Phi_sizes and F_sizes
    dctr = PFN(input_dim=input_dim, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
               Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, latent_dropout=dropout,
               F_dropouts=dropout, summary=summary, optimizer=adam,
               loss=loss) 
    
    
    # sets up keras checkpoints with monitoring of given metric. monitors 'val_loss' with mode 'min' by default 
    checkpoint = tf.keras.callbacks.ModelCheckpoint(savePath + saveLabel + '.h5',
                                                    monitor = monitor,
                                                    verbose = 2,
                                                    save_best_only = True,
                                                    mode = mode)
    
    # sets up CSV Logging of callbacks
    CSVLogger = tf.keras.callbacks.CSVLogger(savePath + saveLabel + '_loss.csv', append=False)
    
    # sets up eraly stopping with given patience (default 15)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = monitor,
                                                     min_delta = 0,
                                                     patience = patience,
                                                     verbose = 1,
                                                     restore_best_weights = True)
    
    learn_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    
    callbacks = [checkpoint, CSVLogger, EarlyStopping, learn_schedule]
    
    return dctr, callbacks


def train(dctr, callbacks, X_train, Y_train, X_val, Y_val, wgt_train=1.0, wgt_val=1.0, 
          epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training'):
    '''
    method to train the given dctr Neural Network with the X_train/Y_train arrays and validate the predictions with X_val and Y_val
    allows for passing along sample_weights for training and validation. These can be positive and/or negative. If no wgt_train or wgt_val are given, then the weights are set to 1 by default
    plots and saves a figure of loss and accuracy throughout the Epochs
    '''

    # check weights
    if isinstance(wgt_train, np.ndarray) == False:
        wgt_train = np.array([wgt_train]*len(X_train))
        
    if isinstance(wgt_val, np.ndarray) == False:
        wgt_val = np.array([wgt_val]*len(X_val))

    print('Starting training')
    history = dctr.fit(X_train, Y_train,
                       sample_weight = pd.Series(wgt_train).to_frame('w_t'), # pd.Series makes the training initialize much, much faster than passing just the weight
                       epochs = epochs,
                       batch_size = batch_size,
                       validation_data = (X_val, Y_val, pd.Series(wgt_val).to_frame('w_v')),
                       verbose = 1,
                       callbacks = callbacks)
                       

    dctr.save(savePath+saveLabel+'.h5')
    
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

    plt.savefig(savePath+saveLabel+'_history.pdf')
    plt.show()


def predict_weights(dctr, X0, X1, batch_size=8192, clip=0.0001):
    '''
    generates weights for reweighing X0 to X1: weights_0
                  and for reweighing X1 to X0: weights_1
    from the predictions made by DCTR
    and returns the reweighing arrays
    '''
    predics_0 = np.clip(dctr.predict(X0, batch_size=batch_size), 0+clip, 1-clip)
    predics_1 = np.clip(dctr.predict(X1, batch_size=batch_size), 0+clip, 1-clip)
    
    weights_0 = predics_0[:,1]/(1-predics_0[:,1])
    weights_1 = predics_1[:,0]/(1-predics_1[:,0])
    
    return weights_0, weights_1


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
from matplotlib import rc
mplhep.style.use('CMS')

rc('text', usetex=True)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

plot_style_0 = {'histtype':'step', 'color':'black', 'linewidth':2, 'linestyle':'--', 'density':True}
plot_style_1 = {'alpha':0.5, 'density':True}

# define dicts of arguments and particles
# [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
# [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]

particles = {0: r'$t\bar{t}$ pair', 
             1: r'$t$',
             2: r'$\bar{t}$'} 

args_dict = {0: r'$p_{T}$ [GeV]',
                 1: r'$\gamma$ rapidity',
                 2: r'$\phi$',
                 3: r'mass [GeV]',
                 4: r'$\eta$ pseudorapidity',
                 5: r'Energy [GeV]',
                 6: r'PID'}

# all particles array
def plot_3_ratio(X0, X1, X2, arg_index, X0_label = 'X0 POWHEG hvq gen', X1_label = 'X1 MiNNLO gen', X2_label = 'POWHEG reweighted', title = '',
                  part_index = 0, X0_wgt = 1,  X1_wgt = 1, X2_wgt = 1, start = -1.2, stop = 1.2, div = 35, ratio_ylim=[0.9,1.1]):
        
    # check weights
    if isinstance(X0_wgt, np.ndarray) == False:
        X0_wgt = [X0_wgt]*len(X0)
    if isinstance(X1_wgt, np.ndarray) == False:
        X1_wgt = [X1_wgt]*len(X1)
    if isinstance(X2_wgt, np.ndarray) == False:
        X2_wgt = [X2_wgt]*len(X2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,10))
    fig.tight_layout(pad=2)

    bins = np.linspace(start, stop, div)
    n0, bins, patches = ax1.hist(X0[:,part_index, arg_index], bins = bins, weights = X0_wgt, label = X0_label, **plot_style_1, color='orange')
    n1, bins, patches = ax1.hist(X1[:,part_index, arg_index], bins = bins, weights = X1_wgt, label = X1_label, **plot_style_1, color='blue')
    n2, bins, patches = ax1.hist(X2[:,part_index, arg_index], bins = bins, weights = X2_wgt, label = X2_label, **plot_style_0)
    
    
    ax1.set_title(str(particles[part_index])+ ': ' + str(args_dict[arg_index]) +' ' +title)
    ax1.set_xlabel(str(args_dict[arg_index])+'['+str(particles[part_index])+']')
    ax1.set_ylabel('probability density')
    ax1.set_xlim([start, stop])
    ax1.legend()
    
    n0 = np.append(n0, n0[-1])
    n1 = np.append(n1, n1[-1])
    n2 = np.append(n2, n2[-1])
    
    num_evts = 0
    tot_evts = len(X2)
    for i in range(tot_evts):
        if (start <= X0[i, 0, arg_index]<= stop):
            num_evts += 1
    
    print(f'{num_evts}out of {tot_evts} events ({np.round(100*num_evts/tot_evts, decimals=2)}%) are between {args_dict[arg_index]} {start} and {stop}')
    
    width = abs(stop - start)/div
    num = n2 * num_evts * width
    denom = n1 * num_evts * width
    
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num/denom
        ratio_variance = num * np.power(denom, -2.0)
        ratio_uncert = np.abs(intervals.poisson_interval(ratio, ratio_variance) - ratio)
    
    ax2.set_title('ratio plot')
    ax2.step(bins, (n0/n1), alpha=0.5, color='orange', where='post')
    ax2.fill_between(bins, n0/n1, 1, label = X0_label, alpha=0.5, color='orange', step='post')
    
    ax2.step(bins, (n2/n1), alpha=0.5, color='green', where='post')
    ax2.fill_between(bins, n2/n1, 1, label = X2_label, alpha=0.5, color='green', step='post')
    
    ax2.hlines(1,start, stop, color='black', linewidth=3, linestyles='dashed', label=str(X1_label)+' baseline')
    ax2.fill_between(bins, ratio_uncert[0,]+1, 1-ratio_uncert[1,], label = 'statistical uncertainty',step='post', alpha=0.2, color='red')
    
    
    ax2.set_ylim(ratio_ylim)
    ax2.set_xlim([start, stop])
    ax2.set_ylabel('ratio')
    ax2.legend()
    
    plt.show()
   

def plot_2(X0, X1, arg_index, X0_label = 'X0 POWHEG hvq gen', X1_label = 'X1 MiNNLO gen', title = '',
            part_index = 0, X0_wgt = 1,  X1_wgt = 1, start = -1.2, stop = 1.2, div = 35):
    
    # check weights
    if isinstance(X0_wgt, np.ndarray) == False:
        X0_wgt = [X0_wgt]*len(X0)
    if isinstance(X1_wgt, np.ndarray) == False:
        X1_wgt = [X1_wgt]*len(X1)
    
    fig, ax1 = plt.subplots(nrows=1, figsize=(7,7))
    
    bins = np.linspace(start, stop, div)
    ax1.hist(X0[:,part_index, arg_index], bins = bins, weights = X0_wgt, label = X0_label, **plot_style_1, color='orange')
    ax1.hist(X1[:,part_index, arg_index], bins = bins, weights = X1_wgt, label = X1_label, **plot_style_1, color='blue')
    
    ax1.set_title(str(particles[part_index])+ ': ' + str(args_dict[arg_index])+' ' +title)
    ax1.set_xlabel(str(args_dict[arg_index])+'['+str(particles[part_index])+']')
    ax1.set_ylabel('probability density')
    ax1.set_xlim([start, stop])
    ax1.legend()
    
    plt.show()


def plot_3(X0, X1, X2, arg_index, X0_label = 'X0 POWHEG hvq gen', X1_label = 'X1 MiNNLO gen', X2_label = 'POWHEG reweighted', title = '',
            part_index = 0, X0_wgt = 1,  X1_wgt = 1, X2_wgt = 1, start = -1.2, stop = 1.2, div = 35):
    
    # check weights
    if isinstance(X0_wgt, np.ndarray) == False:
        X0_wgt = [X0_wgt]*len(X0)
    if isinstance(X1_wgt, np.ndarray) == False:
        X1_wgt = [X1_wgt]*len(X1)
    if isinstance(X2_wgt, np.ndarray) == False:
        X2_wgt = [X2_wgt]*len(X2)
        
    fig, ax1 = plt.subplots(nrows=1, figsize=(7,7))
    
    bins = np.linspace(start, stop, div)
    ax1.hist(X0[:,part_index, arg_index], bins = bins, weights = X0_wgt, label = X0_label, **plot_style_1, color='orange')
    ax1.hist(X1[:,part_index, arg_index], bins = bins, weights = X1_wgt, label = X1_label, **plot_style_1, color='blue')
    ax1.hist(X2[:,part_index, arg_index], bins = bins, weights = X2_wgt, label = X2_label, **plot_style_0)
    
    ax1.set_title(str(particles[part_index])+ ': ' + str(args_dict[arg_index])+' ' +title)
    ax1.set_xlabel(str(args_dict[arg_index])+'['+str(particles[part_index])+']')
    ax1.set_ylabel('probability density')
    ax1.set_xlim([start, stop])
    ax1.legend()
    
    plt.show()
 
    
    
def plot_weights(wgt_0, wgt_1, start = 0, stop = 2, div = 21, title = ''):
    bins = np.linspace(start, stop, div)
    plt.figure(figsize=(4,4))
    
    plt.hist(np.clip(wgt_0, start, stop), bins = bins, label = 'weights 0', color='orange', density =True, alpha=0.5)
    plt.hist(np.clip(wgt_1, start, stop), bins = bins, label = 'weights 1', color='green', density =True, alpha=0.5)

    plt.title('predicted reweighing weights '+title)
    plt.xlabel('')
    plt.ylabel('probability density')
    plt.xlim([start, stop])
    plt.legend()
    #plt.yscale('log')
    plt.show()
    
