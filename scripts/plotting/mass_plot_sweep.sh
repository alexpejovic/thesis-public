#!/usr/bin/env bash

source .env
cd ../../

for DATE in "${MODEL_DATES[@]}"; do
    ../.venv/bin/python scripts/plotting/plot_vae_trainloss.py --model_date ${DATE}
done
