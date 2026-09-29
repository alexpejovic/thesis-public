#!/usr/bin/env bash

source .env
cd ../../

for DATE in "${MODEL_DATES[@]}"; do
    ../.venv/bin/python scripts/plotting/compare_encoder_to_mse2d.py --encoder_date ${DATE}
done
