#!/bin/bash
# file name: submit_DCTR_notebook_training_showered_regular_epochs.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

PROCESS_ID=$(($@ + 1))

python ./DCTR_notebook_training_showered_regular_epochs.py --pid $PROCESS_ID
