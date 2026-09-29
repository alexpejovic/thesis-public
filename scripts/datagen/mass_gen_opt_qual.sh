#!/usr/bin/env bash

DATES=('2026_08_21_12_30_32_353896' '2026_09_01_15_27_29_582021' '2026_09_11_13_32_14_142220' '2026_09_11_13_31_43_098751')

cd ../..

../.venv/bin/python scripts/datagen/generate_encoder_optimization_quality.py --feat_loss mse
../.venv/bin/python scripts/datagen/generate_encoder_optimization_quality.py --feat_loss mse --optimizer jaxley_polyak

for date in "${DATES[@]}"; do
    ../.venv/bin/python scripts/datagen/generate_encoder_optimization_quality.py --encoder_datestr $date
    ../.venv/bin/python scripts/datagen/generate_encoder_optimization_quality.py --encoder_datestr $date --optimizer jaxley_polyak
done
