#!/usr/bin/env bash

source .env
cd ../../

for DATE in "${MODEL_DATES[@]}"; do
    MODEL_DIR="${MODEL_TARGET_DIR}${DATE}"
    ionice -c2 -n5 rsync --exclude 'models' -e "$SSH_COMMAND" --info=progress2 -r $MODEL_DIR $UPLOAD_SERVER_IP:~/$MODEL_SOURCE_DIR
done
