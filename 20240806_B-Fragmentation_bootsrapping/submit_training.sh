#!/bin/bash
# file name: submit_training.sh

source /nfs/dust/cms/user/puschman/tf_venv/bin/activate

PROCESS_ID=$(($@ + 1))

# 5 iterations of training is default
python ./Rb_bootstrapping_training.py -p $PROCESS_ID
