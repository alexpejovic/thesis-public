#!/usr/bin/env bash

source .env
cd ../../

mkdir -p $RECONS_TARGET_DIR

for DATE in "${MODEL_DATES[@]}"; do
    RECON_DIR="${RECONS_SOURCE_DIR}${DATE}"
    ionice -c2 -n5 rsync --exclude 'models' -e "$SSH_COMMAND" --info=progress2 -r $SERVER_IP:~/$RECON_DIR $RECONS_TARGET_DIR
done
