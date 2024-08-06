#!/bin/bash
# file name: submit_MiNNLO.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

SEED=$(($@ + 1))

# Rb = 0.855
python ./generate_events.py 0.855 $SEED 1000000

# Rb = 1.056
python ./generate_events.py 1.056 $SEED 1000000
