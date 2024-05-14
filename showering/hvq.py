#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import standard numerical modules
import numpy as np
import math
import tensorflow as tf
import sys



# In[2]:


data_dir='/nfs/dust/cms/user/puschman/DCTR/git/Data'

path_to_DCTR = '/afs/desy.de/user/v/vaguglie/valentina_venv/Finn_MiNNLO/repeat_Finn_13TeV_hisModel/' # modify as needed
sys.path.append(path_to_DCTR)
import DCTR
from importlib import reload
reload(DCTR)


#hvw before shower (LHE)
X0_test = []
X0_test = DCTR.load_dataset('/nfs/dust/cms/user/vaguglie/converterLHEfiles/MiNNLO_rew/hvq/13TeV_v2/seed100_converted_lhe_new_shower.npz')[:1000000] # 9553938 num of MiNNLO samples

#hvq after shower (after pythia)
X0_plt = []
X0_plt = DCTR.load_dataset('/nfs/dust/cms/user/vaguglie/generators/pythia8307/examples/ShowerTT.npz')[:1000000] # 9553938 num of MiNNLO samples
print('POWHEG hvq all particles X0_plt.shape: '+str(X0_plt.shape))

#### same selcetion as LHE files, i.e. the pseudorapidity selctions
X0_top = DCTR.load_dataset('/nfs/dust/cms/user/vaguglie/generators/pythia8307/examples/ShowerTop.npz')

X0_plt_wgt = X0_test[:, 0, 7].copy()


# setup args for plotting
args = [(X0_plt, X0_plt_wgt, 'hvq after shower'),
        (X0_test, X0_plt_wgt, 'hvq bef shower')]


arg_indices = [0, 3, 4, 5]
part_indices = [0, 1]

## pt(ttbar)
log_bins = np.logspace(np.log10(1), np.log10(1e3), 50)
DCTR.plot_ratio_cms(args, arg_index=0, part_index=0, ratio_ylim=[0.8,1.2], bins=log_bins, y_scale='log') # , ratio_ylim=[0.8, 1.2])
## m(ttbar)
## pT(top)
