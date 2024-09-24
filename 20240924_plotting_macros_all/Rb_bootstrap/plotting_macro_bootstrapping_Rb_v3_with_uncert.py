# coding: utf-8

# standard library imports
from __future__ import absolute_import, division, print_function
import os
import sys
import argparse
import gc

# standard numerical library imports
import math
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import rc
import mplhep as hep

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

sys.path.append('../')
import DCTR


# Global plot settings
import matplotlib.patheffects as pe
from matplotlib import rc
import matplotlib.font_manager
import mplhep as hep
plt.style.use(hep.style.CMS)

pythia_text = r'$PYTHIA \; pp \to  t\bar{t}$'
hep_text = 'Simulation Preliminary'

from matplotlib.lines import Line2D

def make_legend(ax, title, loc='best', font_size = 20):
    leg = ax.legend(frameon=False, loc=loc)
    leg.set_title(title, prop={"size": f'{int(font_size)}'})
    for i, _ in enumerate(leg.texts):
        leg.texts[i].set_fontsize(int(font_size))
    leg._legend_box.align = "left"
    plt.tight_layout()


target_color  = 'black'
rwgt_color    = '#e42536' # red
nominal_color = '#5790fc' # light blue

plt_style_10a = {'color':'#5790fc', 'linewidth':3, 'linestyle':'--'} #, 'density':True, 'histtype':'step'}
plt_style_11a = {'color':'black', 'linewidth':3, 'linestyle':'-'} #', 'density':True, 'histtype':'step'}
plt_style_12a = {'color':'#e42536', 'linewidth':3, 'linestyle':':'} #, 'density':True, 'histtype':'step'}
plt_style_13a = {'color':'blue', 'linewidth':3, 'linestyle':'--'} #, 'density':True, 'histtype':'step'}

target_plt_style  = plt_style_11a
rwgt_plt_style    = plt_style_12a
nominal_plt_style = plt_style_10a


# define plotting functions

# covariance matrix using weighted samples
def weighted_cov(data, weights):
    # print(f'{np.shape(data) = }')

    # Calculate weighted mean
    weighted_mean = np.average(data, axis=0, weights=weights)
    # print(f'{np.shape(weighted_mean) = }')
    # print(f'{weighted_mean = }')

    # Calculate centered data
    centered_data = data - weighted_mean
    # print(f'{np.shape(centered_data) = }')

    # Calculate weighted covariance matrix
    vars = len(data[0,:]) # num of variables per sample
    weighted_covariance = np.zeros(shape=(vars, vars))
    for i in range(vars):
        for j in range(vars):
            weighted_covariance[i, j] = np.sum(
                weights * centered_data[:,i] * centered_data[:,j]) / np.sum(weights)

    return weighted_covariance
    

# correlation matrix using weighted samples
def weighted_corr(data, weights):
    cov = weighted_cov(data, weights)

    # Calculate diagonal matrix for standard deviations
    std_dev = np.sqrt(np.diag(cov))
    std_dev_matrix = np.outer(std_dev, std_dev)

    # Calculate correlation matrix
    weighted_correlation = cov / std_dev_matrix

    return weighted_correlation
    

def plot_matrix(matrix, title, variable_labels = None, vmin=None, vmax=None, savefig = './plots/up/matrix.pdf'):
    font = {'size'   : 14}
    rc('font', **font)

    assert len(matrix[0,:]) == len(matrix[:,0]), 'matrix is not square'
    
    if variable_labels is not None:
        assert len(variable_labels) == len(matrix[0, :]), 'length of variable_labels does not match the dimension of the square matrix'
    else:
        variable_labels = np.arange(1, len(matrix) + 1) # Labels for variables is bin number
    
    figsize = (max(8, (matrix.shape[1]+1)*0.9), max(7, matrix.shape[0])*0.9)
    plt.figure(figsize=figsize)
    
    if vmin is not None and vmax is not None:
        plt.matshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax, fignum=1, aspect='auto')  # Set colormap limits if specified
    else:
        plt.matshow(matrix, cmap='viridis', fignum=1, aspect='auto')  # Default behavior with automatic colormap limits

    plt.colorbar()  # Add colorbar to show scale
    plt.title(title)  # Set the title of the plot
    
    # Add annotations to show values in the heatmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='black',  path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Label x-axis with variable names
    plt.xticks(ticks=np.arange(len(variable_labels)), labels=variable_labels)
    plt.xlabel('bin number')  # Label x-axis

    # Label y-axis with variable names
    plt.yticks(ticks=np.arange(len(variable_labels)), labels=variable_labels)
    plt.ylabel('bin number')  # Label y-axis

    plt.savefig(savefig)


def calc_hists_from_wgts_list(X, bins, wgts_list):
    hist_list = []
    uncert_nrm_list = []
    for i, wgt in enumerate(wgts_list):
        # main density hist
        hist, _ = np.histogram(X, weights = wgt, bins = bins, density=True)
        hist_list.append(hist)

        # non density hist (as in counts) for calculating uncert
        n   , _ = np.histogram(X, weights = wgt, bins = bins, density=False)

        # which bin did each event end up in
        bin_indices = np.digitize(X, bins = bins)

        # sqrt of sum of wgt**2 in each bin
        uncert = np.array([np.sqrt(np.sum(wgt[bin_indices == bin_index]**2)) for bin_index in range(1, len(bins))])
        uncert_nrm = np.divide(uncert, n, out=np.ones_like(uncert), where=(n != 0)) # normalizing uncert by dividing by bin counts, for non-zero bin counts

        uncert_nrm_list.append(uncert_nrm)
    
    return np.array(hist_list), np.array(uncert_nrm_list)



def plot_bootstrapping_from_hists(in_hists, ratio_ylim, obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '', save_folder = './plots_macro', save_prefix = 'plot', legend_loc = 'best'):
    [
        [hist_list, bins],
        [target_hist,  target_uncert ], 
        [nominal_hist, nominal_uncert],
        [mean_hist,    mean_uncert, std_hist]
    ] = in_hists

    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, figsize=(8,10), gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout(pad=1)

    # First subplot
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    axes[0].step(bins, target_hist, label = r'$r_{b}=1.056$', where='post', **target_plt_style)
    axes[0].step(bins, nominal_hist, label = r'$r_{b}=0.855$', where='post', **nominal_plt_style)
    axes[0].step(bins, mean_hist, label = r'$r_{b}=0.855 \; wgt.$ (mean)', where='post', **rwgt_plt_style)

    # Calculate the ratios of histograms
    ratio_0 = target_hist / target_hist
    ratio_1 = mean_hist / target_hist
    ratio_std = std_hist / mean_hist
    ratio_2 = nominal_hist / target_hist

    make_legend(axes[0], pythia_text, loc=legend_loc)

    start = bins[0]
    stop = bins[-1]


    # Constructing the label using Python string formatting
    label = r'$1$/$\sigma \frac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)

    axes[0].set_ylabel(label)
    # axes[0].set_yscale('log')
    axes[0].grid(True)
    axes[0].legend()


    # Second subplot
    axes[1].plot([start, stop], [1,1], '-', color=target_color,  linewidth=3, label=r'$r_{b}=1.056$')
    for hist in hist_list:
        axes[1].plot(bin_centers, hist/target_hist[:-1], linewidth=1, alpha = 0.3, color = 'grey')
    axes[1].plot(bin_centers, ratio_2[:-1], label=r'$r_{b}=0.855$', **nominal_plt_style)
    axes[1].plot(bin_centers, ratio_1[:-1], label = r'$r_{b}=0.855 \; wgt.$ (mean)', **rwgt_plt_style)
    axes[1].fill_between(bin_centers, (ratio_1*(1+ratio_std))[:-1], (ratio_1*(1-ratio_std))[:-1], color=rwgt_color, alpha = 0.5, label = r'$r_{b}=0.855 \; wgt.$ (std)') 

    axes[1].errorbar(bin_centers, ratio_0[:-1], yerr=target_uncert[:-1], fmt='-', color=target_color)
    axes[1].errorbar(bin_centers, ratio_1[:-1], yerr=mean_uncert[:-1], fmt='--', color=rwgt_color)
    axes[1].errorbar(bin_centers, ratio_2[:-1], yerr=nominal_uncert[:-1], fmt=':', color=nominal_color)


    axes[1].set_xlabel(fr'${obs}({part}){unit}$')
    axes[1].set_ylabel(f'Ratio')
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
    axes[0].text(1.0, 1.05, '(13 TeV)', ha="right", va="top", fontsize=20, transform=axes[0].transAxes)

    # adjust strings for named files
    obs = obs.replace('\\', '').replace('{', '').replace('}', '').replace('_','').replace(' ','-').lower()
    part = part.replace('\\', '').replace('{', '').replace('}', '').replace('bar', '').replace(' ','').lower()

    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f'{save_folder}/{save_prefix}_{part}_{obs}_hist.pdf')


def plot_bootstrapping_from_hists_ratio_only(in_hists, ratio_ylim, obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '', save_folder = './plots_macro', save_prefix = 'plot', legend_loc = 'best'):
    [
        [hist_list, bins],
        [target_hist,  target_uncert ], 
        [nominal_hist, nominal_uncert],
        [mean_hist,    mean_uncert, std_hist]
    ] = in_hists

    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=1, figsize=(8,5))
    fig.tight_layout(pad=1)

    # First subplot
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Calculate the ratios of histograms
    ratio_0 = target_hist / target_hist
    ratio_1 = mean_hist / target_hist
    ratio_std = std_hist / mean_hist
    ratio_2 = nominal_hist / target_hist

    make_legend(axes, pythia_text, loc=legend_loc)

    start = bins[0]
    stop = bins[-1]

    # Constructing the label using Python string formatting
    label = r'$1$/$\sigma \frac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)

    # Second subplot
    axes.plot([start, stop], [1,1], '-', color=target_color,  linewidth=3, label=r'$r_{b}=1.056$')
    for hist in hist_list:
        axes.plot(bin_centers, hist/target_hist[:-1], linewidth=1, alpha = 0.3, color = 'grey')
    axes.plot(bin_centers, ratio_2[:-1], label=r'$r_{b}=0.855$', **nominal_plt_style)
    axes.plot(bin_centers, ratio_1[:-1], label = r'$r_{b}=0.855 \; wgt.$ (mean)', **rwgt_plt_style)
    axes.fill_between(bin_centers, (ratio_1*(1+ratio_std))[:-1], (ratio_1*(1-ratio_std))[:-1], color=rwgt_color, alpha = 0.5, label = r'$r_{b}=0.855 \; wgt.$ (std)') 

    axes.errorbar(bin_centers, ratio_0[:-1], yerr=target_uncert[:-1], fmt='-', color=target_color)
    axes.errorbar(bin_centers, ratio_1[:-1], yerr=mean_uncert[:-1], fmt='--', color=rwgt_color)
    axes.errorbar(bin_centers, ratio_2[:-1], yerr=nominal_uncert[:-1], fmt=':', color=nominal_color)


    axes.set_xlabel(fr'${obs}({part}){unit}$')
    axes.set_ylabel(f'Ratio')
    axes.grid(True)


    # print(f'uncertainty NLO: {uncert_nrm_list[0]}')
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
    axes.set_ylim(ratio_ylim)

    axes.set_xlim([start,stop])
    axes.legend(fontsize=13)

    #hep.cms.label(ax=axes[0], data=False, paper=False, lumi=None, fontsize=20, loc=0)
    hep.cms.text(hep_text, loc=0, fontsize=20, ax=axes)
    axes.text(1.0, 1.10, '(13 TeV)', ha="right", va="top", fontsize=20, transform=axes.transAxes)

    # adjust strings for named files
    obs = obs.replace('\\', '').replace('{', '').replace('}', '').replace('_','').replace(' ','-').lower()
    part = part.replace('\\', '').replace('{', '').replace('}', '').replace('bar', '').replace(' ','').lower()

    plt.tight_layout()

    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f'{save_folder}/{save_prefix}_{part}_{obs}_ratio_only_hist.pdf')


# CAUTION !!

# X0 is target
# X1 is nominal
# Opposite to MiNNLO reweighting


# plotting

# 11 bin Xb

# load previously calculated hists from disk
out_hists = np.load('./plots/out_hists_Xb_11bins.npy', allow_pickle=True)
[
    [hist_list, bins],
    [target_hist,  target_uncert ], 
    [nominal_hist, nominal_uncert],
    [mean_hist,    mean_uncert, std_hist]
] = out_hists

# corr_matrix
corr_matrix = weighted_corr(hist_list, weights = [1]*len(hist_list))
plot_matrix(corr_matrix, title = r'x$_b$ correlation between bins', savefig = './plots/xb_11bin_1M_50iter_corr_matrix.pdf')

# hist
plot_bootstrapping_from_hists(out_hists, ratio_ylim=[0.9, 1.15], obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '', save_prefix = '11bins')

plot_bootstrapping_from_hists_ratio_only(out_hists, ratio_ylim=[0.9, 1.15], obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '', save_prefix = '11bins')


# original: 50 bin Xb
# load previously calculated hists from disk
out_hists = np.load('./plots/out_hists_Xb_50bins.npy', allow_pickle=True)
[
    [hist_list, bins],
    [target_hist,  target_uncert ], 
    [nominal_hist, nominal_uncert],
    [mean_hist,    mean_uncert, std_hist]
] = out_hists

# corr_matrix
corr_matrix = weighted_corr(hist_list, weights = [1]*len(hist_list))
plot_matrix(corr_matrix, title = r'x$_b$ correlation between bins', savefig = './plots/xb_50bin_1M_50iter_corr_matrix.pdf')

# hist
plot_bootstrapping_from_hists(out_hists, ratio_ylim=[0.9, 1.15], obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '',  save_prefix = '50bins')
plot_bootstrapping_from_hists_ratio_only(out_hists, ratio_ylim=[0.9, 1.15], obs = r'x_{b}', part = r't\bar{t}', unit = '', inv_unit = '',  save_prefix = '50bins')




# ### B-Quark pT

# pT 10 bins
# load previously calculated hists from disk
out_hists = np.load('./plots/out_hists_b_pt_10bins.npy', allow_pickle=True)
[
    [hist_list, bins],
    [target_hist,  target_uncert ], 
    [nominal_hist, nominal_uncert],
    [mean_hist,    mean_uncert, std_hist]
] = out_hists

# corr matrix
corr_matrix = weighted_corr(hist_list, weights = [1]*len(hist_list))
plot_matrix(corr_matrix, title = r'p$_T$(b) correlation between bins', savefig = './plots/b_pt_10bin_1M_50iter_corr_matrix.pdf')

# hist
plot_bootstrapping_from_hists(out_hists, ratio_ylim=[0.95, 1.10], obs = r'p_{T}', part = 'b', unit = r' [GeV]', inv_unit = r' [GeV$^{-1}$]',  save_prefix = '10bins')
plot_bootstrapping_from_hists_ratio_only(out_hists, ratio_ylim=[0.95, 1.10], obs = r'p_{T}', part = 'b', unit = r' [GeV]', inv_unit = r' [GeV$^{-1}$]',  save_prefix = '10bins')


# org: pT 50 bins
# load previously calculated hists from disk
out_hists = np.load('./plots/out_hists_b_pt_50bins.npy', allow_pickle=True)
[
    [hist_list, bins],
    [target_hist,  target_uncert ], 
    [nominal_hist, nominal_uncert],
    [mean_hist,    mean_uncert, std_hist]
] = out_hists

# corr matrix
corr_matrix = weighted_corr(hist_list, weights = [1]*len(hist_list))
plot_matrix(corr_matrix, title = r'p$_{T}$(b) correlation between bins', savefig = './plots/pT_b_50bin_1M_50iter_corr_matrix.pdf')

# hist
plot_bootstrapping_from_hists(out_hists, ratio_ylim=[0.95, 1.10], obs = r'p_{T}', part = 'b', unit = r' [GeV]', inv_unit = r' [GeV$^{-1}$]',  save_prefix = '50bins')
plot_bootstrapping_from_hists_ratio_only(out_hists, ratio_ylim=[0.95, 1.10], obs = r'p_{T}', part = 'b', unit = r' [GeV]', inv_unit = r' [GeV$^{-1}$]', save_prefix = '50bins')
