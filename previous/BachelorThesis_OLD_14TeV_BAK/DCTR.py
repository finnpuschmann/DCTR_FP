#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: finn puschmann
"""


# energyflow dependencies import
from __future__ import absolute_import, division, print_function
# import tensorflow.keras.backend as K
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
from math import atan2
import matplotlib.pyplot as plt
# from hist import intervals
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


def norm(X, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
    mean = np.nanmean(X[:,1])
    std = np.nanstd(X[:,1])
    norm = (mean, std, ln)
    X -= mean
    X /= std
    return X, norm



def norm_2(X, Y, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
        Y = np.log(np.clip(Y, a_min = 1e-6, a_max = None))
    mean = np.nanmean(np.concatenate((X[:,1], Y[:,1])))
    std = np.nanstd(np.concatenate((X[:,1], Y[:,1])))
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
    norm_pt = (0.0, 1.0, False)
    norm_rapidity = (0.0, 1.0, False)
    norm_phi = (0.0, 1.0, False)
    norm_mass = (0.0, 1.0, False)
    norm_pseudorapidity = (0.0, 1.0, False)
    norm_e = (0.0, 1.0, False)
    
    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    if pt==True: # 0
        X[...,0], norm_pt = norm(X[...,0], ln=True)
    
    if rapidity==True: # 1
        r_std = np.std(X[:,0,1])
        X[...,1] = np.divide(X[...,1], r_std)
        norm_rapidity = (0.0, r_std, False)
        
    if phi==True: # 2
        X[...,2] = X[...,2]/math.pi
        norm_phi = (0.0, math.pi, False)
    
    if mass==True: # 3
        X[...,3], norm_mass = norm(X[...,3], ln=True)
    
    if pseudorapidity==True: # 4
        pr_std = 4
        X[...,4] = np.divide(X[...,4], pr_std)
        norm_pseudorapidity  = (0.0, pr_std, False)
    
    if energy==True: # 5
        X[...,5], norm_e = norm(X[...,5], ln=True)
    
    if PID==True: # 6
        try: X = remap_pid(X)
        except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                               'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    
    if wgt==True: # 7
            X[...,7] = norm_wgt(X[:,0,7])[:, np.newaxis]
    
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
    
    
    if pt==True: # 0
        X[...,0], Y[...,0], norm_pt = norm_2(X[...,0], Y[...,0], ln=True)
    
    if rapidity==True: # 1
        r_std = np.std(np.concatenate((X[:,0,1], Y[:,0,1])) )
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
        pr_std = 4
        X[...,4] = np.divide(X[...,4], pr_std)
        Y[...,4] = np.divide(Y[...,4], pr_std)
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
            X[...,7] = norm_wgt(X[:,0,7])[:, np.newaxis]
            Y[...,7] = norm_wgt(Y[:,0,7])[:, np.newaxis]
    
    norm_dict = {'pt':             norm_pt,
                 'rapidity':       norm_rapidity,
                 'phi':            norm_phi,
                 'mass':           norm_mass,
                 'pseudorapidity': norm_pseudorapidity,
                 'energy':         norm_e}
    
    return X, Y, norm_dict


def norm_per(X, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
    mean = np.nanmean(X)
    std = np.nanstd(X)
    X -= mean
    if std >= 1e-2: # mostly for mass less gluons not causing divide by zero error
        X /= std
    return X


def norm2_per(X, Y, ln=False):
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
        Y = np.log(np.clip(Y, a_min = 1e-6, a_max = None))
    mean = np.nanmean(np.concatenate((X, Y)))
    std = np.nanstd(np.concatenate((X, Y)))
    X -= mean
    Y -= mean
    if std >= 1e-2: # mostly for massless gluons not causing divide by zero error
        X /= std
        Y /= std
 
    return X, Y


def normalize_data_per_particle(X):

    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    for particle in range(len(X[0,:,0])):
        for arg in range(6):
            if arg == 0 or arg == 3 or arg == 5:
                if particle == 0: # don't use log for nrming top-pair mass
                    X[:,particle, arg] = norm_per(X[:,particle, arg], ln=False)
                else: # use log for pt and E
                    X[:,particle, arg] = norm_per(X[:,particle, arg], ln=True) 
            else:
                X[:,particle, arg] = norm_per(X[:,particle, arg], ln=False)
                
    # wgt
    X[X[:,:,7] > 0, 7] = 1 #  masks positive weights and sets them = 1
    X[X[:,:,7] < 0, 7] = -1 # masks negative weights and sets them = -1
    
    # PID
    try: X = remap_pid(X)
    except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                           'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    return X

def normalize_data2_per_particle(X, Y):

    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    
    for particle in range(len(X[0,:,0])):
        for arg in range(6):
            if arg == 0 or arg == 5: # pt and energy
                if particle == 0: # don't use log for nrming top-pair mass
                    X[:,particle, arg], Y[:,particle, arg] = norm2_per(X[:,particle, arg], Y[:,particle, arg], ln=False)
                else:
                    X[:,particle, arg], Y[:,particle, arg] = norm2_per(X[:,particle, arg], Y[:,particle, arg], ln=True)
            else:
                X[:,particle, arg], Y[:,particle, arg] = norm2_per(X[:,particle, arg], Y[:,particle, arg], ln=False)
    
    # wgt
    X[X[:,:,7] > 0, 7] = 1 #  masks positive weights and sets them = 1
    X[X[:,:,7] < 0, 7] = -1 # masks negative weights and sets them = -1
    
    Y[Y[:,:,7] > 0, 7] = 1 #  masks positive weights and sets them = 1
    Y[Y[:,:,7] < 0, 7] = -1 # masks negative weights and sets them = -1
    
    #PID
    try: 
        X = remap_pid(X)
        Y = remap_pid(Y)
    except KeyError: print('remap PID KeyError intercepted. Maybe the PIDs were already remaped,'+
                           'or you are trying to remap PIDs of a someting other than Quarks or Gluons')
    
    return X, Y


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
    Y0 = X0[:,0,-1] # theta is last parameter
    Y1 = X1[:,0,-1]
    
    # removing theta as it was already used to create the classifier arrays Y0 and Y1
    X0 = np.delete(X0, -1, -1)
    X1 = np.delete(X1, -1, -1)
    
    X = []
    X = np.concatenate((X0, X1))
    
    Y = []
    Y = np.concatenate((Y0, Y1))
    Y = to_categorical(Y, num_classes=2)
    
    class_wgt = 1
    if use_class_weights==True:
        class_wgt=len(X0)/len(X1)
        X0[...,-1] /= class_wgt
    
    # create weights array from dataset
    weights_array = X[:,0,-1] # weigts are last paramter after removing theta
    
    # removing weights as it was already used to create the weights_array
    X = np.delete(X, -1, -1) 
    
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



def setup_nn(input_dim=7, Phi_sizes = (100,100,128), F_sizes = (100,100,100),
             loss = 'cce', dropout=0.0, l2_reg=0.0, Phi_acts='relu', F_acts='relu', output_act='softmax',
             learning_rate=0.001, patience=10, use_scheduler=True, monitor='val_loss', 
             mode='min', savePath=currentPath, saveLabel='DCTR_training', summary=False):
    
    # supported losses
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()

    if loss == 'mse':
        loss = mse_loss
    else:
        loss = cce_loss
    
    # activation functions: if string is unsuppported, fallback to 'relu'
    supported_acts = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
    if Phi_acts not in supported_acts:
        print(f"unrecognized Phi activation '{Phi_acts}', falling back to 'relu'")
        Phi_acts = 'relu'
    if F_acts not in supported_acts:
        print(f"unrecognized F activation '{F_acts}', falling back to 'relu'")
        F_acts = 'relu'
    if output_act not in supported_acts:
        print(f"unrecognized output activation '{F_acts}', falling back to 'softmax'")
        F_acts = 'softmax'
    
    
    # optimizer
    adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # defines dctr as a particle flow network with given paramterization
    dctr = PFN(input_dim=input_dim, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
               Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, latent_dropout=dropout,
               F_dropouts=dropout, summary=summary, optimizer=adam,
               loss=loss, Phi_acts=Phi_acts, F_acts=F_acts) 
    
    
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
    
    callbacks = [checkpoint, CSVLogger, EarlyStopping, learn_schedule]
    
    return dctr, callbacks


def train(dctr, callbacks, X_train, Y_train, X_val, Y_val, wgt_train=1.0, wgt_val=1.0, 
          epochs=80, batch_size=8192, savePath=currentPath, saveLabel='DCTR_training'):
    '''
    method to train the given dctr Neural Network with the X_train/Y_train arrays and validate the predictions with X_val and Y_val
    allows for passing along sample_weights for training and validation. These can be positive and/or negative. If no wgt_train or wgt_val are given, then the weights are set to 1 by default
    plots and saves a figure of loss and accuracy throughout the Epochs
    '''
    
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


def predict_weights(dctr, X0, X1, batch_size=8192, clip=0.00001, verbose=1):
    '''
    generates weights for reweighing X0 to X1: weights_0
                  and for reweighing X1 to X0: weights_1
    from the predictions made by DCTR
    and returns the reweighing arrays
    '''
    predics_0 = np.clip(dctr.predict(X0, batch_size=batch_size, verbose=verbose), 0+clip, 1-clip)
    predics_1 = np.clip(dctr.predict(X1, batch_size=batch_size, verbose=verbose), 0+clip, 1-clip)
    
    weights_0 = predics_0[:,1]/(1-predics_0[:,1])
    weights_1 = predics_1[:,0]/(1-predics_1[:,0])
    
    weights_0 /= np.mean(weights_0) # adjust weights so that mean is 1
    weights_1 /= np.mean(weights_1) # adjust weights so that mean is 1
    
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
rc('font', size=14)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)

# define dicts of arguments and particles
# [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
# [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]

particles = {0: r'$t\bar{t}$ pair', 
             1: r'$t$',
             2: r'$\bar{t}$'} 

args_dict = {0: r'$p_{T}$ [GeV]',
             1: r'$y$ rapidity',
             2: r'$\phi$',
             3: r'mass [GeV]',
             4: r'$\eta$ pseudorapidity',
             5: r'Energy [GeV]',
             6: r'PID'}


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
               bins = None, optimal_bins = False, start = None, stop = None, div = 35, 
               ratio_ylim=[0.9,1.1], figsize=(8,8), layout='rows', stats_only=False):
    
    # binning: prio: passed bins, calculated optimal bins, linear bins from start, stop, div
    if bins is not None: 
        bins = bins
    elif optimal_bins == True: # 2. prio: calculate optimal bins
        if args[0][0].ndim > 1: # check whether full array
            bins = np.histogram_bin_edges(args[0][0][:,part_index, arg_index], bins = 'auto')
        else:
            bins = np.histogram_bin_edges(args[0][0], bins = 'auto')
    else: # no passed bins, nor optimal bins
        if start is None: # was startstop given?
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
    
    start = bins[0]
    stop = bins[-1]
    div = len(bins)
    # width = bins[1] - bins [0]
    
    if stats_only == False:
        if layout == 'cols':
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
        else: fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
        fig.tight_layout(pad=2)
    
    n_list = [] # list of histogram bin counts
    n_sum_list = [] # list of total counts in all bins: used for normalizing
    uncert_list = [] # list of uncertainties sqrt(n) for each bin in each histogram
    mae_list = []
    chi2_list = []
    p_list = []
    
    for i, (X, wgt, label) in enumerate(args):
        # check wheter full dataset is passed or 1D dataset, that can be plotted as is
        if X.ndim > 1:
            n, bins = np.histogram(X[:,part_index, arg_index], bins = bins, weights = wgt)
            bin_indices = np.digitize(X[:,part_index, arg_index], bins = bins)
        else: 
            n, bins = np.histogram(X, bins = bins, weights = wgt)
            bin_indices = np.digitize(X, bins = bins)
        
        # statistics
        # uncert: sqrt of the square of all weights in each bin
        uncert = np.array([np.sqrt(np.sum(wgt[bin_indices == bin_index]**2)) for bin_index in range(1, len(bins))])
        uncert_list.append(uncert)
        uncert_nrm = uncert/n
        uncert_nrm = np.append(uncert_nrm, uncert_nrm[-1]) # extend list by last element for plotting
        
        # normalize so that all counts are equal to first passed X
        n_sum = np.sum(n)
        n_sum_list.append(n_sum)
        n *= (n_sum_list[0] / n_sum)
        n_list.append(n)
        # calculate MAE statistics and chi^2
        mae = np.nanmean(np.absolute(n_list[0] - n))
        chi2 = np.nansum(np.power(n_list[0] - n, 2)/(np.power(uncert, 2) + np.power(uncert_list[0], 2)))
        p = stats.chi2.sf(chi2, len(bins) - 2)
        print(f'{label}: \n Mean Absolute Error {mae} \n chi square of {chi2} with p {p} \n compared to {args[0][2]}')
        mae_list.append(mae)
        chi2_list.append(chi2)
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
            hist = np.append(n, n[-1])
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
            ax1.set_ylabel(f'counts normalized to {args[0][2]}')
        else: ax1.set_ylabel(y_label)
        
        ax1.set_xlim([start, stop])
        ax1.set_ylim(bottom=0)
        ax1.legend()
        
        ax2.set_title('ratio plot')
        ax2.set_xlim([start, stop])
        ax2.set_ylim(ratio_ylim)
        ax2.set_ylabel('ratio')
        ax2.legend()
        
        plt.show()
    
    return mae_list, chi2_list, p_list

