#!/usr/bin/env bash
# Train the beta ladder at additional training seeds, so the MSE-distance study can
# average over VAE training variance instead of reporting one draw per beta.
#
# Everything except --beta and --seed comes from configs/vae.yaml, whose defaults
# already match the five encoders trained on 2026-08-17 (TimeVAEBase, latent_dim 64,
# gelu, ZScoreGlobal, masking off, 25 epochs, lr 8e-4, batch 128, conv [50,100,200]).
# Overriding nothing else is what keeps the new runs comparable to those five.
#
# SEED 5678 is deliberately NOT here -- it is already trained. Run from
# slurm_scripts/:
#
#   ./mass_train_beta_ladder.sh            # submit
#   ./mass_train_beta_ladder.sh --dry-run  # print what would be submitted
#
# ~4 h per run measured from the 2026-08-17 ladder. 5 betas x 3 seeds = 15 jobs.
#
# NOT a study spec, deliberately. train_vae.py declares `trace_data` as a
# POSITIONAL argument (train_vae.py:319) and study.argv_for emits flags only, so
# study.py cannot drive this target without changing that CLI. At 15 jobs a
# per-value loop is still within what CLAUDE.md permits; anything larger needs
# the positional turned into a flag first.
#
# The authoritative ladder definition is configs/beta_mse.yaml (ladder_betas,
# ladder_train_seeds) -- that is what the scoring script filters encoders
# against. These defaults must agree with it; override them here only for a
# one-off.

set -euo pipefail

# shellcheck disable=SC2206  # word splitting is how the override is passed
BETAS=(${BETAS:-0.0 0.05 0.1 0.15 0.2})
SEEDS=(${SEEDS:-1234 4321 9876})
DATASET=${DATASET:-Pospischil}

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="echo [dry-run]"; fi

for seed in "${SEEDS[@]}"; do
    for beta in "${BETAS[@]}"; do
        $DRY sbatch ./train_vae.sh --beta "$beta" --seed "$seed" "$DATASET"
    done
done

echo "submitted ${#BETAS[@]} betas x ${#SEEDS[@]} seeds = $((${#BETAS[@]} * ${#SEEDS[@]})) jobs"
