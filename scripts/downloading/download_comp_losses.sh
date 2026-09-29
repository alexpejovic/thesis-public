#!/usr/bin/env bash

source .env
cd ../../

mkdir -p $COMPLOSS_LOCAL_DIR

rsync -avh -e "$SSH_COMMAND" --info=progress2 --ignore-existing $SERVER_IP:~/$COMPLOSS_SERVER_DIR $COMPLOSS_LOCAL_DIR
