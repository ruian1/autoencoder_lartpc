#!/bin/bash

#export TOP=/scratch/ruian/maskrcnn_run
export DLLEE_UNIFIED_BASEDIR=/usr/local/share/dllee_unified
source /usr/local/bin/thisroot.sh

cd /usr/local/share/dllee_unified
source configure.sh

source /scratch/ruian/autoencoder_lartpc/setup.sh
source /scratch/ruian/autoencoder_lartpc/uboone/LArCV/configure.sh

cd /scratch/ruian/autoencoder_lartpc/uboone/

nohup python train_autoencoder_2.py autoencoder_2.cfg > train.txt &

