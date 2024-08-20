#!/bin/bash
# file name: submit_hvq_dileptonic_HadronLevel-Off_MPI-Off_FSR-Off.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

LHE=$(($@ + 1))

# min pt 15
python ./jet_multiplicity_hvq_dileptonic_HadronLevel-Off_MPI-Off_FSR-Off.py -l $LHE
