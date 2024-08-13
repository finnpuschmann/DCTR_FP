#!/bin/bash
# file name: submit_hvq_dileptonic.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

LHE=$(($@ + 1))

# min pt 15
python ./jet_multiplicity_hvq_dileptonic.py -p 15.0 -l $LHE

