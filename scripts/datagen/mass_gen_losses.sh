#!/usr/bin/env bash

source .env
cd ../..

for DATE in "${DATES[@]}"; do
    ../.venv/bin/python ./scripts/datagen/generate_losses.py --encoder_datestr ${DATE} &
    # sleep 3m
done
