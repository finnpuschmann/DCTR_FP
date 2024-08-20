#!/bin/bash
# file name: submit_DCTR_training_neural_positive.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

PROCESS_ID=$(($@ + 1))

# 5 iterations of training is default
python ./submit_DCTR_training_neural_positive_MSE_vs_BCE.py -p $PROCESS_ID
