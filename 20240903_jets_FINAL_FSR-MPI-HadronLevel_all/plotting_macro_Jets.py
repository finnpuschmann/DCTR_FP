import os

from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# all functions and methods defined here are also defined in DCTR.py
# defined again here, so that this script can run with only
# matplotlib, numpy, copy and mplhep as dependencies
# plotting settings and help functions

def make_legend(ax, title, loc='best', font_size = 20):
    leg = ax.legend(frameon=False, loc=loc)
    leg.set_title(title, prop={"size": f'{int(font_size)}'})
    for i, _ in enumerate(leg.texts):
        leg.texts[i].set_fontsize(int(font_size))
    leg._legend_box.align = "left"
    plt.tight_layout()


def plot_ratio_cms_from_hists(
        in_hists, label_list, arg_index = 0, part_index = 0, loc0 = 'best', loc1 = 'best', font_size = 20,
        ratio_ylim=[0.9,1.1], pythia_text = r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA', ylim_min = None, ylim_max = None,
        figsize=(8,10), y_scale=None, hep_text = 'Simulation Preliminary', center_mass_energy = '(13 TeV)',
        part_label=None, arg_label=None, unit=None, inv_unit=None, save_prefix = 'plot', save_folder = None):

    try:
        n_list, uncert_nrm_list, bin_edges = in_hists
    except:
        print('in_hists not in right form. Needs to be in_hists = [n_list, uncert_nrm_list, bin_edges]')
        return

    # make sure each hist has a label
    assert len(label_list) == len(n_list), 'differnt number of labels and histograms!'

    # different order then plot_ratio_cms() functions, for easier looping # swapped 10a and 11a

    plt_style_10a = {'color':'black',   'linestyle':'-' }
    plt_style_11a = {'color':'#5790fc',   'linestyle':'--'}
    plt_style_12a = {'color':'#e42536', 'linestyle':':' }
    plt_style_13a = {'color':'blue',    'linestyle':':' }


    # binning: prio: passed bins, calculated bins from quantiles, linear bins from start, stop, div
    start = copy(bin_edges[0])
    stop = copy(bin_edges[-1])

    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    fig.tight_layout(pad=1)

    # First subplot
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

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
    make_legend(axes[0], pythia_text, loc0, int(font_size))

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
    #label = r'$1$/$\sigma \frac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)
    label = r'$1 / \sigma \dfrac{d\sigma}{d %s(%s)}$ %s' % (obs, part, inv_unit)
    axes[0].set_ylabel(label)

    if y_scale == 'log':
        axes[0].set_yscale('log')
    else:
        axes[0].set_ylim(bottom=0)
    axes[0].grid(True)

    if ylim_min is not None:
        axes[0].set_ylim(bottom=ylim_min)
    if ylim_max is not None:
        axes[0].set_ylim(top=ylim_max)

    # Second subplot
    for i, label in enumerate(label_list):
        if label == 'NLO (hvq) wgt.':
            label = 'Reweighted NNLO'
        if i == 0:
            axes[1].errorbar(bin_centers, ratio_list[i][:-1], yerr=uncert_nrm_list[i][:-1], linewidth=1.5, **plt_style_10a)
            axes[1].plot([start, stop], [1,1], label=label, linewidth=2, **plt_style_10a)
        elif i % 3 == 0:
            axes[1].errorbar(bin_centers, ratio_list[i][:-1], yerr=uncert_nrm_list[i][:-1], linewidth=1.5, **plt_style_13a)
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = label, linewidth=2, **plt_style_13a)
        elif i % 2 == 0:
            axes[1].errorbar(bin_centers, ratio_list[i][:-1], yerr=uncert_nrm_list[i][:-1], linewidth=1.5, **plt_style_12a)
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = label, linewidth=2, **plt_style_12a)
        else:
            axes[1].errorbar(bin_centers, ratio_list[i][:-1], yerr=uncert_nrm_list[i][:-1], linewidth=1.5, **plt_style_11a)
            axes[1].plot(bin_centers, ratio_list[i][:-1], label = label, linewidth=2, **plt_style_11a)


    axes[1].set_xlabel(fr'${obs}({part}){unit}$')
    axes[1].set_ylabel('Ratio(/NNLO)')
    axes[1].grid(True)

    # print(f'uncertainty NLO: {uncert_nrm_list[0]}')
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95)
    axes[1].set_ylim(ratio_ylim)

    axes[0].set_xlim([start,stop])
    axes[1].set_xlim([start,stop])
    axes[1].legend(fontsize=13, loc = loc1)

    #hep.cms.label(ax=axes[0], data=False, paper=False, lumi=None, fontsize=20, loc=0)
    hep.cms.text(hep_text, loc=0, fontsize=20, ax=axes[0])
    axes[0].text(1.0, 1.05, center_mass_energy, ha="right", va="top", fontsize=20, transform=axes[0].transAxes)

    # Save the figure
    if save_folder == None:
        if part_index == 0:
            save_folder = './plots/tt-pair'
        elif part_index == 1:
            save_folder = './plots/top'
        else:
            save_folder = './plots/anti-top'
    # make save_folder directory, if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # adjust strings for named files
    obs = obs.replace('\\', '').replace('{', '').replace('}', '').replace(' ','-').lower()
    part = part.replace('\\', '').replace('{', '').replace('}', '').replace('bar', 'anti').lower()

    plt.savefig(f'{save_folder}/{save_prefix}_{obs}_{part}.pdf')
    # plt.show()


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
    label_list = ['NNLO (MiNNLO)', 'NLO (hvq)',  'NLO (hvq) wgt.']
    # hists saved to disk are in form:
    # hists = [hist_list, uncert_nrm_list, bin_edges]
    # where each _list contains an entry for each label in label_list

    # each plot is in tt-pair subfolder

    # multiplicity
    name = 'plot_multiplicity'
    hists = np.load(f'./plots/tt-pair/{name}_jet_histograms.npy', allow_pickle=True)

    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.7, 1.3],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='Jet', arg_label='multiplicity', 
        unit='', inv_unit='')

    # Jets WITH b jet
    # 1st jet
    name = 'with_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')

    name = 'with_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')
    
    name = 'with_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')



    # 2nd jet
    name = 'with_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')

    name = 'with_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')
    
    name = 'with_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')



    # 3rd jet
    name = 'with_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')
    
    name = 'with_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')
    
    name = 'with_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')



    # Jets WITHOUT b jet
    # 1st jet
    name = 'no_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')

    name = 'no_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')
    
    name = 'no_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_1stjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='1st Jet')


    # 2nd jet
    name = 'no_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')

    name = 'no_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')
    
    name = 'no_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_2ndjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='2nd Jet')


    # 3rd jet
    name = 'no_b_jets_15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')

    name = 'no_b_jets_20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')
    
    name = 'no_b_jets_25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_3rdjet_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='3rd Jet')



    # b-jets
    name = '15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_bjets_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='B Jets')

    name = '20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_bjets_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='B Jets')
    
    name = '25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_bjets_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.6, 1.4],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='B Jets')


    # b-quarks
    name = '15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_b_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='b')

    name = '20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_b_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='b')
    
    name = '25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_b_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='b')


    # W Bosons
    name = '15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_w_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='W')

    name = '20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_w_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='W')
    
    name = '25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_w_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='W')


    # leptons
    name = '15bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_leptons_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='leptons')

    name = '20bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_leptons_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='leptons')
    
    name = '25bins'
    hists = np.load(f'./plots/tt-pair/{name}_pt_leptons_histograms.npy', allow_pickle=True)
    plot_ratio_cms_from_hists(hists, label_list,
        save_folder = './plotting_macro',
        y_scale='log', ratio_ylim=[0.8, 1.2],
        pythia_text=r'$POWHEG \; pp \to  t\bar{t}$ + PYTHIA',
        save_prefix = name, part_label='leptons')
    
