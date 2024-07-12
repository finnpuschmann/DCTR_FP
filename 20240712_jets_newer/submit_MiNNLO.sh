#!/bin/bash
# file name: submit_MiNNLO.sh

cd ~
source pythia_env.sh

# min pt 15
python ./pythia8309/examples/jet_jet_multiplicity_MiNNLO.py -p 15.0 -l $@

# min pt 50
python ./pythia8309/examples/jet_jet_multiplicity_MiNNLO.py -p 50.0 -l $@