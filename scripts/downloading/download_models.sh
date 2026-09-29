#!/usr/bin/env bash

source .env
cd ../../

mkdir -p $MODEL_TARGET_DIR

for DATE in "${MODEL_DATES[@]}"; do
    MODEL_DIR="${MODEL_SOURCE_DIR}${DATE}"
    ionice -c2 -n5 rsync --exclude 'models' -e "$SSH_COMMAND" --info=progress2 -r $SERVER_IP:~/$MODEL_DIR $MODEL_TARGET_DIR
done
