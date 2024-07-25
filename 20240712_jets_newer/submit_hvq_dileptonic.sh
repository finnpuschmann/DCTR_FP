#!/bin/bash
# file name: submit_hvq_dileptonic.sh

source pythia_env.sh

# min pt 15
python jet_multiplicity_hvq_dileptonic.py -p 15.0

# min pt 50
python jet_multiplicity_hvq_dileptonic.py -p 50.0