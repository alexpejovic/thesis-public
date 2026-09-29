#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=20         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-01:00            # Runtime in D-HH:MM
#SBATCH --partition=a100-galvani     # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/%j-region_lr_sweep.out  # File to which STDOUT will be written
#SBATCH --error=logs/%j-region_lr_sweep.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=alex.pejovic@student.uni-tuebingen.de  # Email to which notifications will be sent

# Calibrate the two optimizer arms that opt_region_arms.yaml is about to run.
#
# Why this exists: adam's lr = 0.05 was measured (job 2799826) while rmsprop's
# 0.01 was a guess, and an uncalibrated arm entering a "which optimizer is
# best" comparison is a confound, not a result. Every arm has to be tuned by
# the same rule -- smallest median BEST-iterate distance among the learning
# rates that keep their runs inside the box -- which is what
# gen_region_success_scores.py --lr_report applies. It reproduces adam's 0.05
# on the 2026-08-28 sweep, which is the check that the rule is the same one.
#
# jaxley-polyak is swept too, but as a DIAGNOSTIC only: its step coefficient
# stays at the reference loop's literal 1/3 (todo.md:21) whatever the sweep
# says, because the brief for that arm is that it behave exactly like the loop.
# The sweep is here so that "1/3 is not where the optimum is" would at least be
# on the record rather than unknown.
#
#   cd encoding/slurm_scripts && sbatch ./region_lr_sweep.sh

set -uo pipefail

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi
echo -e "---------------------------------\n"

cd ..

PY=../.venv/bin/python
OUT=./data/tuning/optimizer-lr-sweep.jsonl

# One cell, held fixed across every lr so the comparison is paired: the same
# parameter triplet, the same loss, the same starts. Small on purpose (3 ideal
# x 64 init against the study's 10 x 100) -- this is a step-size question, not
# a score, and it has to fit in one short job.
SET_KEY="${SET_KEY:-gLeak+eLeak+gNa}"
LOSS="${LOSS:-encoder-zscore-mask}"
IDEAL="${IDEAL:-3}"
INIT="${INIT:-64}"

RMSPROP_LRS="${RMSPROP_LRS:-0.003 0.01 0.03 0.1}"
# 1/6, 1/3, 2/3 -- half, the reference value, and double.
JAXLEY_LRS="${JAXLEY_LRS:-0.16667 0.33333 0.66667}"

sweep () {   # sweep <optimizer> <lrs...>
    local opt="$1"; shift
    echo "-------- ${opt} LR SWEEP --------"
    for LR in "$@"; do
        $PY ./scripts/datagen/gen_region_success_scores.py \
            --set_key "$SET_KEY" --loss "$LOSS" \
            --optimizer "$opt" --max_learning_rate "$LR" \
            --ideal_seeds 0 "$IDEAL" --init_seeds 0 "$INIT" \
            --out "$OUT" --no-save_raw
    done
    echo "---------------------------------"
}

sweep rmsprop $RMSPROP_LRS
sweep jaxley-polyak $JAXLEY_LRS

echo "-------- REPORT -----------------"
# Non-zero if no learning rate in a sweep is admissible: that is a result about
# the range, and the job must not exit 0 having established nothing.
$PY ./scripts/datagen/gen_region_success_scores.py --lr_report "$OUT"
RC=$?
echo "---------------------------------"
echo "lr_report rc=$RC"
exit $RC
