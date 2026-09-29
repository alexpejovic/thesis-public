#!/usr/bin/env bash

source .env
cd ../../

mkdir -p $OPT_SWEEP_LOCAL_DIR

rsync -avh -e "$SSH_COMMAND" --info=progress2 --ignore-existing $SERVER_IP:~/$OPT_SWEEP_SERVER_DIR $OPT_SWEEP_LOCAL_DIR
