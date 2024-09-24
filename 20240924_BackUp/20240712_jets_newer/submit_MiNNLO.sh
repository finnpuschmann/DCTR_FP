#!/bin/bash
# file name: submit_MiNNLO.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

LHE=$(($@ + 1))

# min pt 15
python ./jet_multiplicity_MiNNLO.py -p 15.0 -l $LHE
