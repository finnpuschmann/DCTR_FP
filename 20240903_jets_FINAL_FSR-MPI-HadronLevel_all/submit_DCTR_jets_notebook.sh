#!/bin/bash
# file name: submit_DCTR_jets_notebook.sh

# source pythia and add examples folder to PATH (for Makefile and lib)
source /nfs/dust/cms/user/puschman/pythia_env.sh

export PATH=/nfs/dust/cms/user/puschman/pythia8309/examples:$PATH
export PATH=/nfs/dust/cms/user/puschman/pythia8309/lib:$PATH

python ./DCTR_jets_notebook_20240917_b-filtered_my-samples.py

echo 'finished calculating hists, starting plotting macro'

python ./plotting_macro_Jets.py
