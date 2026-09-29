#!/usr/bin/env bash

source .env
cd ../../

# DIR_TO_UPLOAD="data/models/vae/Pospischil-small"
# DIR_TARGET="thesis/encoding/data/models/vae/"
DIR_TO_UPLOAD="data/voltage_traces/Pospischil_feats_mask.npz"
DIR_TARGET="thesis/encoding/data/voltage_traces/"

# ionice -c2 -n5 rsync -e "$SSH_COMMAND" --info=progress2 -r $DIR_TO_UPLOAD $SERVER_IP:~/$DIR_TARGET
ionice -c2 -n5 rsync -e "$SSH_COMMAND" --info=progress2 $DIR_TO_UPLOAD $SERVER_IP:~/$DIR_TARGET
