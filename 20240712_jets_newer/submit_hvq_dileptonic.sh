#!/bin/bash
# file name: submit_hvq_dileptonic.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

if [ "$1" -eq 0 ]; then
    python jet_multiplicity_hvq_dileptonic.py -p 15
elif [ "$1" -eq 1 ]; then
    python jet_multiplicity_hvq_dileptonic.py -p 30
elif [ "$1" -eq 2 ]; then
    python jet_multiplicity_hvq_dileptonic.py -p 50
else
    echo "Invalid argument. Please provide 0, 1, or 2."
fi
