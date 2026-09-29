#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <channel_type>"
    exit 1
fi

source .env
cd ../../

mkdir -p $TRACE_LOCAL_DIR

DATA_DIR="${TRACE_SERVER_DIR}${1}.dat.new"
FEATS_DIR="${TRACE_SERVER_DIR}${1}_feats.npz.new"
AMPS_DIR="${TRACE_SERVER_DIR}${1}_amps.npz.new"
PARAMS_DIR="${TRACE_SERVER_DIR}${1}_params.npz.new"
rsync -e "$SSH_COMMAND" --info=progress2 $SERVER_IP:~/$DATA_DIR $TRACE_LOCAL_DIR
rsync -e "$SSH_COMMAND" --info=progress2 $SERVER_IP:~/$FEATS_DIR $TRACE_LOCAL_DIR
rsync -e "$SSH_COMMAND" --info=progress2 $SERVER_IP:~/$AMPS_DIR $TRACE_LOCAL_DIR
rsync -e "$SSH_COMMAND" --info=progress2 $SERVER_IP:~/$PARAMS_DIR $TRACE_LOCAL_DIR
