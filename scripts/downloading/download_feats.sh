#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <channel_type>"
    exit 1
fi

source .env
cd ../../

mkdir -p $TRACE_LOCAL_DIR

FEAT_DIR="${TRACE_SERVER_DIR}${1}_feats.npz"
rsync -e "$SSH_COMMAND" --info=progress2 $SERVER_IP:~/$FEAT_DIR $TRACE_LOCAL_DIR
