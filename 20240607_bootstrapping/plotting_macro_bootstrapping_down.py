import os

from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import mplhep as hep
import matplotlib.patheffects as pe

# all functions and methods defined here are also defined in DCTR.py
# defined again here, so that this script can run with only
# matplotlib, numpy, copy and mplhep as dependencies

def make_legend(ax, title):
    leg = ax.legend(frameon=False)
    leg.set_title(title, prop={'size':20})
    leg._legend_box.align = "left"
    plt.tight_layout()

def plot_ratio_bootstrapped(in_hists, hist_list, label_list, 
        ratio_ylim=[0.80, 1.20], denominator = 'Up', 
        pythia_text = r'$POWHEG \; pp \to t\bar{t}$',
        hep_text = 'Simulation Preliminary', 
        save_prefix = '40M_50iter'):

    try:
        n_list, ratio_std, bin_edges = in_hists
    except:
        print('in_hists not in right form. Needs to be in_hists = [n_list, ratio_std, bin_edges]')
        return

    # make sure each hist has a label
    assert len(label_list) == len(n_list), 'differnt number of labels and histograms!'

    # different order then plot_ratio_cms() functions, for easier looping # swapped 10a and 11a
    plt_style_10a = {'color':'black',   'linestyle':'-' }
    plt_style_11a = {'color':'Green',   'linestyle':'--'}
    plt_style_12a = {'color':'#FC5A50', 'linestyle':':' }
    plt_style_13a = {'color':'blue',    'linestyle':':' }

    font = {'size':16}
    rc('font', **font)

    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, figsize=(8,11), gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout(pad=1)

    # First subplot
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    start = bin_edges[0]
    stop  = bin_edges[-1]

    ratio_list = []

    for i, label in enumerate(label_list):
        if i == 0:
            axes[0].step(bin_edges, n_list[i], label = label, where='post', linewidth=3, **plt_style_10a)
        elif i % 3 == 0:
            axes[0].step(bin_edges, n_list[i], label = label, where='post', linewidth=3, **plt_style_13a)
        elif i % 2 == 0:
            axes[0].step(bin_edges, n_list[i], label = label, where='post', linewidth=3, **plt_style_12a)
        else:
            axes[0].step(bin_edges, n_list[i], label = label, where='post', linewidth=3, **plt_style_11a)

        # Calculate the ratios of histograms
        ratio = n_list[i] / n_list[0] # comparing to first passed hist
        ratio_list.append(ratio)


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
    for i, label in enumerate(label_list):
        if i == 0:       # 0 | target
            axes[1].plot([start, stop], [1,1], label=label, linewidth=2, **plt_style_10a)
        elif i % 3 == 0: # 3, 6, 9, ...
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = label, linewidth=2, **plt_style_13a)
        elif i % 2 == 0: # 2, 4, 8, ...
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = label, linewidth=2, **plt_style_12a)
        elif i == 1:     # 1 meann
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = f'{label} + (mean)', linewidth=2, **plt_style_11a)
            axes[1].fill_between(bin_centers, (ratio_list[i]*(1+ratio_std))[:-1], (ratio_list[i]*(1-ratio_std))[:-1], alpha = 0.6, label = f'{label} (std)', **plt_style_11a) 
    
    # plot all resulting hists in hist_list faintly
    for hist in hist_list:
        axes[1].plot(bin_centers, hist/target_hist[:-1], linewidth=1, alpha = 0.3, color = 'grey')
    
    axes[1].set_xlabel(fr'${obs}({part}){unit}$')
    axes[1].set_ylabel(f'Ratio(/{denominator})')
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
    if part_index == 0:
        save_folder = './plots/tt-pair'
    elif part_index == 1:
        save_folder = './plots/top'
    else:
        save_folder = './plots/anti-top'
    # make save_folder directory, if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # adjust strings for named files
    obs = obs.replace('\\', '').replace('{', '').replace('}', '').replace('_','').replace(' ','-').lower()
    part = part.replace('\\', '').replace('{', '').replace('}', '').replace('bar', '').lower()
    denominator = denominator.lower()

    plt.savefig(f'{save_folder}/{denominator}/{save_prefix}_{obs}_{part}.pdf')
    # plt.show()


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


def plot_square_matrix_heatmap(matrix, title, variable_labels = None, vmin=None, vmax=None, savefig = './plots/up/matrix.pdf'):
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
    plt.show()


# plot hists
if __name__ == '__main__':
    # cms plot style
    plt.style.use(hep.style.CMS)
    
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

    # setup args for plotting
    label_list = ['NNLO (MiNNLO)', 'NLO (hvq)',  'Reweighted NNLO']
    # hists saved to disk are in form:
    # hists = [hist_list, uncert_nrm_list, bin_edges]
    # where each _list contains an entry for each label in label_list


    # TODO !!!
    # Adjust for Bootstrapping plots, including corr matrices





    # EXAMPLE from plotting macro for regular epochs
    # top plots
    # p_t(t) log binning
    name = 'showered_log_0800_31'
    hists_pt_top = np.load(f'./plots/top/{name}_pt_t_histograms.npy', allow_pickle=True)

    plot_ratio_cms_from_hists(hists_pt_top, label_list,
        arg_index=0, part_index=1,
        y_scale='log', ratio_ylim=[0.85, 1.15],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name)

    