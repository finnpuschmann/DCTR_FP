# import system modules
import sys
import os
import gc
import argparse

# import standard numerical modules
import numpy as np
import math

import matplotlib.pyplot as plt

# import machine learning modules
import tensorflow as tf
import keras.backend as K

sys.path.append('../')
import DCTR


# # Process Data
data_dir = '../../Data'
num_events = 9686913 # = num MiNNLO events


# ## hvq dielptonic

# LOAD W Bosons, leptons and no_b_jets from disk

X0_jets_no_b = []

X0_W = []
X0_lep = []

X0 = []
X0_nJets = []
X0_jet = []
X0_b_jets = []
X0_b_quarks = []

data_dir_hvq = f'{data_dir}/POWHEG_hvq/dileptonic/20240917'

for i in range(1, 1001):
    X0.extend(
        np.load(f'{data_dir_hvq}/converted_lhe_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )

    X0_nJets.extend(
        np.load(f'{data_dir_hvq}/jet_multiplicity_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_jet.extend(
        np.load(f'{data_dir_hvq}/jet_4vectors_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_b_jets.extend(
        np.load(f'{data_dir_hvq}/b_jets_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_b_quarks.extend(
        np.load(f'{data_dir_hvq}/b_quarks_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_W.extend(
        np.load(f'{data_dir_hvq}/W_Bosons_array_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_lep.extend(
        np.load(f'{data_dir_hvq}/leptons_array_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    X0_jets_no_b.extend(
        np.load(f'{data_dir_hvq}/jet_4vectors_no_b_hvq_dileptonic_10K_{i}_b-filtered.npy')
    )
    if i % 50 == 0:
        print(f'read {i} files')

X0 = np.array(X0)
print(X0.shape)

# nrm data and save to disk (only needed to run once)

nrm_array_dir = f'{data_dir}/POWHEG_hvq/showered/'
nrm_array = np.load(f'{nrm_array_dir}/norm_array_lhe_01.npy')
X0_nrm, _ = DCTR.normalize_data(X0.copy(), nrm_array)
X0_nrm = np.array(X0_nrm)
np.save(f'{data_dir_hvq}/normed_lhe_hvq_dileptonic_10K_1-1000_b-filtered.npy', X0_nrm)

X0       = np.array(X0[:num_events])

X0_W   = np.array(X0_W[:num_events])
X0_lep = np.array(X0_lep[:num_events])

X0_jets_no_b = np.array(X0_jets_no_b[:num_events])

X0_nJets = np.array(X0_nJets[:num_events])
X0_jet   = np.array(X0_jet[:num_events])

X0_b_jets    = np.array(X0_b_jets[:num_events])
X0_b_quarks  = np.array(X0_b_quarks[:num_events])

X0_nrm = np.load(f'{data_dir_hvq}/normed_lhe_hvq_dileptonic_10K_1-1000_b-filtered.npy')[:num_events]
X0_wgt = X0_nJets[:,1]

print(X0.shape)
print(X0_nrm.shape)
print(X0_nJets.shape)
print(X0_jet.shape)


# ## MiNNLO
gc.collect()

# MiNNLO has 10k events per lhe
data_dir_minnlo = f'{data_dir}/MiNNLO/showered/20240917'

X1 = []
X1_nJets = []
X1_jet = []

X1_jets_no_b = []
X1_W = []
X1_lep = []

X1_b_jets = []
X1_b_quarks = []

for i in range(1, 1001):
    X1.extend(
        np.load(f'{data_dir_minnlo}/converted_lhe_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_nJets.extend(
        np.load(f'{data_dir_minnlo}/jet_multiplicity_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_jet.extend(
        np.load(f'{data_dir_minnlo}/jet_4vectors_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_b_jets.extend(
        np.load(f'{data_dir_minnlo}/b_jets_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_b_quarks.extend(
        np.load(f'{data_dir_minnlo}/b_quarks_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_W.extend(
        np.load(f'{data_dir_minnlo}/W_Bosons_array_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_lep.extend(
        np.load(f'{data_dir_minnlo}/leptons_array_MiNNLO_10K_{i}_b-filtered.npy')
    )
    X1_jets_no_b.extend(
        np.load(f'{data_dir_minnlo}/jet_4vectors_no_b_MiNNLO_10K_{i}_b-filtered.npy')
    )
    if i % 50 == 0:
        print(f'read {i} files')

X1 = np.array(X1)
print(X1.shape)

# nrm data and save to disk (only needed to run once)

nrm_array_dir = f'{data_dir}/POWHEG_hvq/showered/'
nrm_array = np.load(f'{nrm_array_dir}/norm_array_lhe_01.npy')
X1_nrm, _ = DCTR.normalize_data(X1.copy(), nrm_array)
np.save(f'{data_dir_minnlo}/normed_lhe_MiNNLO_10K_1-1000_b-filtered.npy', X1_nrm)

X1       = np.array(X1[:num_events])
print(X1.shape)

X1_W   = np.array(X1_W[:num_events])
X1_lep = np.array(X1_lep[:num_events])
X1_jets_no_b = np.array(X1_jets_no_b[:num_events])

X1_nJets = np.array(X1_nJets[:num_events])
X1_jet   = np.array(X1_jet[:num_events])

X1_b_jets   = np.array(X1_b_jets[:num_events])
X1_b_quarks = np.array(X1_b_quarks[:num_events])

X1_nrm = np.load(f'{data_dir_minnlo}/normed_lhe_MiNNLO_10K_1-1000_b-filtered.npy')[:num_events]
X1_wgt = X1_nJets[:, 1]

print(X1.shape)
print(X1_nrm.shape)
print(X1_nJets.shape)
print(X1_jet.shape)



gc.collect()



print(X0.shape)
print(X0_nrm.shape)
print(X0_nJets.shape)
print(X0_jet.shape)

print(X0_b_jets.shape)
print(X0_b_quarks.shape)


print(X1.shape)
print(X1_nrm.shape)
print(X1_nJets.shape)
print(X1_jet.shape)

print(X1_b_jets.shape)
print(X1_b_quarks.shape)



# delete energy from (normalized) testing dataset, since the neural network was trained without these parameters
X0_nrm = np.delete(X0_nrm, 5, -1) # E
X0_nrm = np.delete(X0_nrm, 4, -1) # eta

# garbage collection after deleting to clear memory asap
print(gc.collect())

print(X0_nrm.shape)
print(X0_nrm[0])

# # DCTR reweighting
# calculate weights from DCTR trained on showered events
model = '../20240521_showered_new/train_20240523_regular_epochs.tf' # path to previously trained model (in .tf format (folder)) for showered events

dctr_rwgt = []
# calculate rwgt
with tf.device('CPU'):
    dctr_rwgt = DCTR.get_rwgt([model], X0_nrm) # .tf models also include network architecture. get_rwgt() sets up the network for the (list of) models, then calls predict_weights() like in DCTR_notebook_OLD_14-to-13TeV notebook


# wgts_plot = [(dctr_rwgt[0], r'DCTR NLO $\to$ NNLO reweights')]
# DCTR.plot_weights(wgts_plot, start = 0.1, stop = 10)

# apply orginal generator weights to rwgt
dctr_rwgt = np.multiply(dctr_rwgt[0], X0_wgt)



# plots


pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA'

# jet multiplicity

args_multiplicity = [(X1_nJets[:,0], X1_wgt, 'Jet Multiplicity NNLO (MiNNLO) \n dileptonic decays'),
                     (X0_nJets[:,0], X0_wgt, 'Jet Multiplicity NLO (hvq) \n dileptonic decays'),
                     (X0_nJets[:,0], dctr_rwgt, r'DCTR reweighted NLO $\to$ NNLO')]

bins = np.linspace(-0.5, 17.5, 19)
ratio_ylim=[0.7, 1.3]

DCTR.plot_ratio_cms(args_multiplicity, bins = bins, y_scale = 'log', ratio_ylim=ratio_ylim, part_label='Jet', arg_label='multiplicity', unit='', inv_unit='', pythia_text=pythia_text)


# all jets pT

# 1st jet
log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_0 = [(X1_jet[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_0 = [(X1_jet[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_0 = [(X1_jet[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_25bins')


# 2nd jet
log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_1 = [(X1_jet[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_1 = [(X1_jet[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_1 = [(X1_jet[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_25bins')


# 3rd jet
log_bins = np.logspace(np.log10(30), np.log10(600), 16)
args_jet_2 = [(X1_jet[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(600), 21)
args_jet_2 = [(X1_jet[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(600), 26)
args_jet_2 = [(X1_jet[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jet[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jet[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'with_b_jets_25bins')



# jet pT (no b_jets)

# 1st jet
log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_0 = [(X1_jets_no_b[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_0 = [(X1_jets_no_b[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_0 = [(X1_jets_no_b[:, 0, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 0, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 0, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='1st Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_25bins')


# 2nd
log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_1 = [(X1_jets_no_b[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_1 = [(X1_jets_no_b[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_1 = [(X1_jets_no_b[:, 1, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 1, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 1, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_1, y_scale = 'log', part_label='2nd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_25bins')


# 3rd
log_bins = np.logspace(np.log10(30), np.log10(600), 16)
args_jet_2 = [(X1_jets_no_b[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_15bins')

log_bins = np.logspace(np.log10(30), np.log10(600), 21)
args_jet_2 = [(X1_jets_no_b[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_20bins')

log_bins = np.logspace(np.log10(30), np.log10(600), 25)
args_jet_2 = [(X1_jets_no_b[:, 2, 0], X1_wgt, 'NNLO (MiNNLO)'),
              (X0_jets_no_b[:, 2, 0], X0_wgt, 'NLO (hvq)'),
              (X0_jets_no_b[:, 2, 0], dctr_rwgt, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_2, y_scale = 'log', part_label='3rd Jet', ratio_ylim=(0.6, 1.4), bins = log_bins, pythia_text=pythia_text, save_prefix = 'no_b_jets_25bins')


# b jets
print(X1_b_jets.shape)

# concat first and second b_jets
X0_b_jets_pt_concat = np.concatenate((X0_b_jets[:,0, 0], X0_b_jets[:,1, 0]))
X1_b_jets_pt_concat = np.concatenate((X1_b_jets[:,0, 0], X1_b_jets[:,1, 0]))

# concat wgt with itself, to have correct wgt for concat b jet array
X0_wgt_concat = np.concatenate((X0_wgt, X0_wgt))
X1_wgt_concat = np.concatenate((X1_wgt, X1_wgt))

dctr_rwgt_concat = np.concatenate((dctr_rwgt, dctr_rwgt))

print(X0_b_jets_pt_concat.shape)
print(X1_b_jets_pt_concat.shape)

print(X0_wgt_concat.shape)
print(X1_wgt_concat.shape)

print(dctr_rwgt_concat.shape)


# b-jet pT
log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_0 = [(X1_b_jets_pt_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_b_jets_pt_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_b_jets_pt_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b Jets', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_0 = [(X1_b_jets_pt_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_b_jets_pt_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_b_jets_pt_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b Jets', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_0 = [(X1_b_jets_pt_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_b_jets_pt_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_b_jets_pt_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b Jets', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '25bins')



# W Bosons and leptons

# concat W Bosons and leptons (like the b jets above)
# W
X0_W_concat = np.concatenate((X0_W[:,0, 0], X0_W[:,1, 0])) # only pt, to save on memory
X1_W_concat = np.concatenate((X1_W[:,0, 0], X1_W[:,1, 0]))

print(X0_W_concat.shape)
print(X1_W_concat.shape)


# lepton
X0_lep_concat = np.concatenate((X0_lep[:,0, 0], X0_lep[:,1, 0])) # only pt, to save on memory
X1_lep_concat = np.concatenate((X1_lep[:,0, 0], X1_lep[:,1, 0]))

print(X0_lep_concat.shape)
print(X1_lep_concat.shape)


# W pT

log_bins = np.logspace(np.log10(1), np.log10(800), 16)
args_jet_0 = [(X1_W_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_W_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_W_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='W', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '15bins')

log_bins = np.logspace(np.log10(1), np.log10(800), 21)
args_jet_0 = [(X1_W_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_W_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_W_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='W', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '20bins')

log_bins = np.logspace(np.log10(1), np.log10(800), 26)
args_jet_0 = [(X1_W_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_W_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_W_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='W', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '25bins')



# lepton pT

log_bins = np.logspace(np.log10(1), np.log10(500), 16)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='leptons', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '15bins')

log_bins = np.logspace(np.log10(1), np.log10(500), 21)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='leptons', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '20bins')

log_bins = np.logspace(np.log10(1), np.log10(500), 26)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='leptons', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '25bins')


# b quark pT

log_bins = np.logspace(np.log10(30), np.log10(800), 16)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '15bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 21)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '20bins')

log_bins = np.logspace(np.log10(30), np.log10(800), 26)
args_jet_0 = [(X1_lep_concat, X1_wgt_concat, 'NNLO (MiNNLO)'),
              (X0_lep_concat, X0_wgt_concat, 'NLO (hvq)'),
              (X0_lep_concat, dctr_rwgt_concat, 'NLO (hvq) wgt.')]

DCTR.plot_ratio_cms(args_jet_0, y_scale = 'log', part_label='b', ratio_ylim=(0.4, 1.6), bins = log_bins, pythia_text=pythia_text, save_prefix = '25bins')



'''

# sum of jet (no b jets) pt per event

X0_jet_pt_sums = []
for event in X0_no_b_jet:
    X0_jet_pt_sums.append(np.sum(event[:,0]))

X1_jet_pt_sums = []
for event in X1_no_b_jet:
    X1_jet_pt_sums.append(np.sum(event[:,0]))

    
print(f'{np.shape(X0_jet_pt_sums) = }')
print(f'{np.shape(X1_jet_pt_sums) = }')


args_jet_sum = [(np.array(X1_jet_pt_sums), X1_wgt, 'NNLO (MiNNLO)'),
                (np.array(X0_jet_pt_sums), X0_wgt, 'NLO (hvq)'),
                (np.array(X0_jet_pt_sums), dctr_rwgt, 'NLO (hvq) wgt.')]

log_bins = np.logspace(np.log10(15), np.log10(1200), 15)
lin_bins = np.linspace(15, 800, 15)

DCTR.plot_ratio_cms(args_jet_sum, bins = log_bins, y_scale = 'log', part_label='Sum of Jets', ratio_ylim=(0.8, 1.2))



# sum of jets (incl b jets) pt per event

X0_jet_pt_sums = []
for event in X0_jet:
    X0_jet_pt_sums.append(np.sum(event[:,0]))

X1_jet_pt_sums = []
for event in X1_jet:
    X1_jet_pt_sums.append(np.sum(event[:,0]))

    
print(f'{np.shape(X0_jet_pt_sums) = }')
print(f'{np.shape(X1_jet_pt_sums) = }')


args_jet_sum = [(np.array(X1_jet_pt_sums), X1_wgt, 'NNLO (MiNNLO)'),
                (np.array(X0_jet_pt_sums), X0_wgt, 'NLO (hvq)'),
                (np.array(X0_jet_pt_sums), dctr_rwgt, 'NLO (hvq) wgt.')]

log_bins = np.logspace(np.log10(15), np.log10(1200), 15)
lin_bins = np.linspace(15, 800, 15)

DCTR.plot_ratio_cms(args_jet_sum, bins = log_bins, y_scale = 'log', part_label='Sum of Jets', ratio_ylim=(0.8, 1.2))
'''
