#!/bin/bash
# file name: submit_hvq_dileptonic_b-filtered.sh

# source pythia and add examples folder to PATH (for Makefile and lib)
source /nfs/dust/cms/user/puschman/pythia_env.sh

export PATH=/nfs/dust/cms/user/puschman/pythia8309/examples:$PATH
export PATH=/nfs/dust/cms/user/puschman/pythia8309/lib:$PATH

LHE=$(($@ + 1))

# min pt 15
python ./jet_multiplicity_hvq_dileptonic_b-filtered.py -p 15.0 -l $LHE

