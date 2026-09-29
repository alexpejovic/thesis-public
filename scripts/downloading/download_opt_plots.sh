#!/usr/bin/env bash

source .env
cd ../../

mkdir -p $OPT_PLOTS_LOCAL_DIR

rsync -avh -e "$SSH_COMMAND" --info=progress2 --ignore-existing $SERVER_IP:~/$OPT_PLOTS_SERVER_DIR $OPT_PLOTS_LOCAL_DIR
