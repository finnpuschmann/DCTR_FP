#!/bin/bash
# file name: submit_hvq_dileptonic.sh

source /nfs/dust/cms/user/puschman/pythia_env.sh

if [$@ == 0]
then
    python jet_multiplicity_hvq_dileptonic.py -p 15
fi

if [$@ == 1]
then
    python jet_multiplicity_hvq_dileptonic.py -p 30
fi

if [$@ == 2]
then
    python jet_multiplicity_hvq_dileptonic.py -p 50
fi
