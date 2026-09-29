#!/usr/bin/env bash
# Pull the beta-vs-MSE distance records back from the cluster.
#
# The records are small json (one per parameter subset), so this is fast; the
# optional *_losses.npz grids from --save_losses runs are ~6 MB per loss and are
# excluded by default. Set BETA_WITH_GRIDS=1 to fetch them too.
#
# Expects in encoding/.env:
#   SSH_COMMAND, SERVER_IP
#   BETA_DIST_SERVER_DIR   e.g. thesis/encoding/data/beta_mse_distance
#   BETA_DIST_LOCAL_DIR    e.g. encoding/data/

set -euo pipefail

source .env
cd ../../

mkdir -p "$BETA_DIST_LOCAL_DIR"

EXCLUDES=(--exclude '*_losses.npz')
if [[ -n "${BETA_WITH_GRIDS:-}" ]]; then
    EXCLUDES=()
fi

rsync -r -e "$SSH_COMMAND" --info=progress2 "${EXCLUDES[@]}" \
    "$SERVER_IP:~/$BETA_DIST_SERVER_DIR" "$BETA_DIST_LOCAL_DIR"
